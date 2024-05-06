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
    Trainer,
    MistralForCausalLM
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

accelerator = Accelerator()
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
    def __init__(self,model,args,tokenizer):
        nn.Module.__init__(self)
        self.args = args
        self.sentence_encoder = model
        self.tokenizer = tokenizer
        self.batch_size = args.train.per_device_train_batch_size
        self.eval_batch_size = args.train.per_device_eval_batch_size
        self.linear = nn.Linear(model.config.hidden_size,2)
        self.dropout = nn.Dropout(0.1)
        self.num_class = 2
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.step_token_id = 32000
    def forward(self,**inputs):
        meta_info = inputs['meta_info']
        input_ids = inputs["input_ids"]
        labels = inputs['labels']
        rewards = self.linear(self.dropout(self.sentence_encoder(input_ids=input_ids)[0]))
        pred_labels = []
        loss = 0
        correct_count=0
        total_count=0
        batch_size = input_ids.shape[0]
        step_pre_scores = []
        loss = self.loss(rewards.view(-1,2),labels.view(-1).long())
        for i in range(batch_size):
            eos_indices = [i for i, x in enumerate(input_ids[i]) if x == self.step_token_id]
            rewards[i] = torch.softmax(rewards[i], -1)
            pred_score, pred_label = torch.max(rewards[i][eos_indices], dim=-1)
            step_label = labels[i][eos_indices]
            pred_score[pred_label == 0] = 1 - pred_score[pred_label == 0]
            step_pre_scores.append(pred_score)
            for j in range(len(meta_info[i]['step_labels'])):
                if pred_label[j] == step_label[j]:
                    correct_count+=1
            total_count+=len(meta_info[i]['step_labels'])
            pred_labels.append(pred_label.detach().cpu())
            meta_info[i]['loss'] = loss.item()
        final_pred_score= torch.tensor([torch.pow(torch.prod(scores),1/len(scores)) for scores in step_pre_scores],device=input_ids.device)
        for i in range(batch_size):
            meta_info[i]['pred_score'] = final_pred_score[i].item()
        loss = loss/batch_size
        return loss,meta_info,correct_count,total_count,pred_labels

def main():
    args = parse_args()
    update_config(cfgs, args)
    mkdir(cfgs.log.output_dir)
    cfgs.seed = 42
    cfgs.is_local_main_process  = accelerator.is_local_main_process

    set_random_seed(cfgs.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfgs.model.model_path,trust_remote_code=True)
    tokenizer.pad_token_id = 0
    config = AutoConfig.from_pretrained(cfgs.model.model_path,trust_remote_code=True)
    # config.pre_seq_len = 10
    # config.prefix_projecion = False
    # config.use_cache = False

    model = AutoModel.from_pretrained(cfgs.model.model_path,trust_remote_code=True,config=config)
    tokenizer.add_special_tokens({'additional_special_tokens':["<step>"]})
    model.resize_token_embeddings(len(tokenizer))
    model = CumstomModel(model,cfgs,tokenizer)
    model.load_state_dict(torch.load(cfgs.evaluator.checkpoint_path))
    model.half()
    test_dataset = CustomDataset_For_Step_Verifiers(cfgs.dataset.test_data_path, 
                                tokenizer,cfgs.dataset.max_src_len, cfgs.dataset.max_seq_len, model_type=cfgs.model.model_type,cfgs=cfgs)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    test_dataloader = DataLoader(test_dataset,
                                  collate_fn=DataCollatorWithPadding_For_Step_Math(tokenizer),
                                  sampler=test_sampler,
                                  batch_size=cfgs.train.per_device_eval_batch_size,
                                  pin_memory=True)

    model= accelerator.prepare_model(model,evaluation_mode=True)
    accelerator.print(f"-------------------Beginning Evaluation-----------------")
    model.eval()
    all_info = []
    pred_labels = []
    ground_truths = []
    with tqdm(enumerate(test_dataloader),total = len(test_dataloader)) as pbar:
        for step,batch in pbar:
            batch = to_device(batch,accelerator.device)
            with torch.no_grad():
                loss,meta_info,correct_count,total_count,pred_label= model.forward(**batch)
                for i in range(len(meta_info)):
                    ground_truths.extend(meta_info[i]['step_labels'])
                    pred_labels.extend(pred_label[i])
                all_info.extend(meta_info)
    ground_truths = gather_objects(ground_truths)
    pred_labels = gather_objects(pred_labels)
    acc = accuracy_score(ground_truths,pred_labels)
    f1 = f1_score(ground_truths,pred_labels)
    recall = recall_score(ground_truths,pred_labels)
    precision = precision_score(ground_truths,pred_labels)
    all_info = gather_objects(all_info)
    datatype_dic={}
    for i in all_info:
        if i['dataset'] not in datatype_dic:
            datatype_dic[i['dataset']]={'loss':[i['loss']]}
        else:
            datatype_dic[i['dataset']]['loss'].append(i['loss'])
    Metrics = {}
    for key in list(datatype_dic.keys()):
            loss = datatype_dic[key]['loss']
            Metrics[key]={'loss':sum(loss),"f1":f1,"recall":recall,'precision':precision,"acc":acc}
            accelerator.print(Metrics)
    if accelerator.is_local_main_process:
        with open(os.path.join(cfgs.dataset.infer_result,cfgs.log.run_name+f"{acc}%.json"),"w",encoding='utf-8') as f:
            json.dump(all_info,f,ensure_ascii=False,indent=4)

if __name__ == "__main__":
    main()