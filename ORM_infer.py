import argparse
import os
import math
import sys
from shutil import copyfile
from utils import *
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModel,
    SchedulerType,
    get_scheduler,
    Trainer
)
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from config import rm_config as cfgs, update_config
from accelerate import Accelerator
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import torch.nn as nn
from tqdm import tqdm
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

accelerator = Accelerator(mixed_precision='bf16')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")

    parser.add_argument(
        '--config_file',
        type=str,
        default='',
        help="Path to training config file"
    )

    parser.add_argument("--local_rank",
                        type=int,
                        default=0,
                        help="local_rank for distributed training on gpus")

    parser.add_argument(
        'opts',
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    return args

class CumstomModel(nn.Module):
    def __init__(self,model,args):
        nn.Module.__init__(self)
        self.args = args
        self.sentence_encoder = model
        self.batch_size = args.train.per_device_train_batch_size
        self.eval_batch_size = args.train.per_device_eval_batch_size
        self.linear = nn.Linear(model.config.hidden_size,2)
        self.dropout = nn.Dropout(0.1)
        self.pad_id = 0
        self.num_class = 2
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)
    def forward(self,**inputs):
        meta_info = inputs['meta_info']
        input_ids = inputs["input_ids"]
        labels = inputs['labels']
        bs = input_ids.size(0)
        sentence_logits = self.linear(self.dropout(self.sentence_encoder(input_ids=input_ids)[0]))
        bs_correct_count = 0
        pred_labels = []
        non_pad_idxs=[]
        for i in range(bs):
            non_pad_idx = max(torch.where(input_ids[i] != self.pad_id)[0])
            non_pad_idxs.append(non_pad_idx)

        loss = torch.tensor(0)
        non_pad_idxs = torch.tensor(non_pad_idxs).to(sentence_logits.device)
        rewards = sentence_logits[torch.arange(bs), non_pad_idxs, :] #bs 2
        
        loss = self.loss(rewards, labels.view(-1))
        rewards = torch.softmax(rewards,-1)
        for i in range(bs):
            if rewards[i][1] > 0.5:
                 pred_label = 1
            else:
                pred_label = 0
            pred_labels.append(pred_label)
            if labels[i]==pred_label:
                meta_info[i]['correctness'] = 1
                bs_correct_count+=1
            else:
                meta_info[i]['correctness'] = 0
            meta_info[i]['loss'] = loss.mean().item()
            meta_info[i]['pred_score'] = rewards[i][1].item()
        return loss,meta_info,bs_correct_count,pred_labels
    
def main():
    args = parse_args()
    update_config(cfgs, args)
    cfgs.seed = 42
    
    set_random_seed(cfgs.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfgs.model.model_path,trust_remote_code=True)
    tokenizer.pad_token_id = 0
    config = AutoConfig.from_pretrained(cfgs.model.model_path,trust_remote_code=True)
    # config.pre_seq_len = 10
    model = AutoModel.from_pretrained(cfgs.model.model_path,trust_remote_code=True,config=config)
    model = CumstomModel(model,cfgs)
    model.load_state_dict(torch.load(cfgs.evaluator.checkpoint_path))
    model.half()
    test_dataset = CustomDataset_For_Verifiers(cfgs.dataset.test_data_path, 
                                tokenizer,cfgs.dataset.max_src_len, cfgs.dataset.max_seq_len, model_type=cfgs.model.model_type,cfgs=cfgs)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    test_dataloader = DataLoader(test_dataset,
                                  collate_fn=DataCollatorWithPadding_For_Math(tokenizer),
                                  sampler=test_sampler,
                                  batch_size=cfgs.train.per_device_eval_batch_size,
                                  pin_memory=True)
    model= accelerator.prepare_model(model)

    accelerator.print(f"-------------------Beginning Evaluation-----------------")

    model.eval()
    all_info = []
    with tqdm(enumerate(test_dataloader),total = len(test_dataloader),disable=not accelerator.is_local_main_process) as pbar:
        for step,batch in pbar:
            batch = to_device(batch, accelerator.device)
            with torch.no_grad():
                loss,meta_info,bs_correct_count,pred_labels= model.forward(**batch)
                all_info.extend(meta_info)

        datatype_dic={}
        for i in all_info:
            if i['dataset'] not in datatype_dic:
                datatype_dic[i['dataset']]={'loss':[i['loss']],'correctness':[i['correctness']],"pred_score":[i['pred_score']],"answer":[i['answer']]}
            else:
                datatype_dic[i['dataset']]['loss'].append(i['loss'])
                datatype_dic[i['dataset']]['correctness'].append(i['correctness'])
                datatype_dic[i['dataset']]['answer'].append(i['answer'])
                datatype_dic[i['dataset']]['pred_score'].append(i['pred_score'])
        Metrics = {}
        for key in list(datatype_dic.keys()):
            correctness = datatype_dic[key]['correctness']
            loss = datatype_dic[key]['loss']
            Metrics[key]={'acc':sum(correctness)/len(correctness),'loss':sum(loss)/len(correctness)}
    shepherd_loss = Metrics.get(cfgs.dataset.data_type,{'loss':0})['loss']
    shepherd_acc = round(Metrics.get(cfgs.dataset.data_type,{"acc":0})['acc']*100,2)
    accelerator.print(f"loss:{shepherd_loss}, acc: {shepherd_acc}")
    all_info = gather_objects(all_info)
    accelerator.print(len(all_info))
    if accelerator.is_local_main_process:
        with open(os.path.join(cfgs.dataset.infer_result,cfgs.log.run_name+f"{shepherd_acc}%.json"),"w",encoding='utf-8') as f:
            json.dump(all_info,f,ensure_ascii=False,indent=4)

if __name__ == '__main__':
    main()