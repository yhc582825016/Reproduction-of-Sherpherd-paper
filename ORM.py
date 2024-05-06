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
    get_scheduler,
    Trainer,
    MistralForCausalLM
)
from config import rm_config as cfgs, update_config
from accelerate import Accelerator
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
        return loss,meta_info,bs_correct_count,pred_labels

def main():
    args = parse_args()
    update_config(cfgs, args)
    mkdir(cfgs.log.output_dir)
    if accelerator.is_main_process:
        init_tracker(cfgs)
        cfg_save_path = os.path.join(cfgs.log.output_dir, os.path.basename(args.config_file))
        copyfile(args.config_file, cfg_save_path)

    set_random_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(cfgs.model.model_path,trust_remote_code=True)
    tokenizer.pad_token_id = 0
    config = AutoConfig.from_pretrained(cfgs.model.model_path,trust_remote_code=True)
    # config.pre_seq_len = 10
    # config.prefix_projecion = False
    # config.use_cache = False

    model = AutoModel.from_pretrained(cfgs.model.model_path,trust_remote_code=True,config=config)
    model = CumstomModel(model,cfgs)
    # optimizer_grouped_parameters = get_optimizer_grouped_parameters(
    # model, cfgs.train.weight_decay)
    # model = model.half()
    # model.sentence_encoder.transformer.prefix_encoder.float()

    optimizer = torch.optim.AdamW(model.parameters(), cfgs.train.learning_rate)
    train_dataset = CustomDataset_For_Verifiers(cfgs.dataset.train_data_path, 
                                tokenizer,cfgs.dataset.max_src_len, cfgs.dataset.max_seq_len, model_type=cfgs.model.model_type,cfgs=cfgs)
    val_dataset = CustomDataset_For_Verifiers(cfgs.dataset.val_data_path, 
                                tokenizer,cfgs.dataset.max_src_len, cfgs.dataset.max_seq_len, model_type=cfgs.model.model_type,cfgs=cfgs)
    train_dataloader, eval_dataloader = create_rm_dataloder_for_math(cfgs, train_dataset, val_dataset, tokenizer)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfgs.train.gradient_accumulation_steps)
    num_update_steps_in_total = int(num_update_steps_per_epoch * cfgs.train.num_train_epochs)
    num_warmup_steps = int(num_update_steps_in_total * 0.1)

    lr_scheduler = get_scheduler(
        name=cfgs.train.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_update_steps_in_total,
    )

    model,lr_scheduler,optimizer,train_dataloader,eval_dataloader= accelerator.prepare(model,lr_scheduler,optimizer,train_dataloader,eval_dataloader)
    global_step = 0
    accelerator.print(f"-------------------Beginning Evaluation------------------")
    Metrics= evaluate_model_for_math(model, eval_dataloader,accelerator.device)
    accelerator.print(f"step:{global_step}, losses:{Metrics[cfgs.dataset.data_type]['loss']}, acc: {round(Metrics[cfgs.dataset.data_type]['acc']*100,2)}")
    eval_stats = {
        'eval/loss':Metrics[cfgs.dataset.data_type]['loss'] ,
        'eval/acc': round(Metrics[cfgs.dataset.data_type]['acc']*100,2) ,
        'eval/F1': round(Metrics[cfgs.dataset.data_type]['f1']*100,2) ,
        'eval/precision': round(Metrics[cfgs.dataset.data_type]['precision']*100,2) ,
        'eval/recall': round(Metrics[cfgs.dataset.data_type]['recall']*100,2) ,
    }
    if accelerator.is_main_process:
        wandb_log(eval_stats, global_step)
        file_path = os.path.join(cfgs.train.save_path,cfgs.model.model_type,cfgs.log.run_name)
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)
    accelerator.print("***** Running training *****")
    best_f1 = 0
    for epoch in range(cfgs.train.num_train_epochs):
        correct_count = 0
        total_count = 0
        accelerator.print(f"Beginning of Epoch {epoch+1}/{cfgs.train.num_train_epochs}, Total Micro Batches {len(train_dataloader)}")
        model.train()
        with tqdm(enumerate(train_dataloader),total = len(train_dataloader),desc=f"current epoch : {epoch}",\
                disable=(not accelerator.is_main_process)) as pbar:
            for step ,batch in pbar:
                batch = to_device(batch, accelerator.device)
                loss,meta_info,bs_correct_count,pred_labels = model.module.forward(**batch)
                correct_count+=bs_correct_count
                total_count+=batch['input_ids'].size(0)
                acc = correct_count/total_count
                accelerator.backward(loss)
                pbar.set_description("loss:%.2f,acc:%.2f%%" % (loss, acc*100))
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                stats = {'train/loss': loss.item(),
                        'train/epoch':epoch,
                        'train/acc':acc,
                        'lr': optimizer.param_groups[0]['lr']}
                global_step += 1
                if accelerator.is_main_process:
                    wandb_log(stats, global_step)
                if global_step % int((cfgs.log.eval_epoch_ratio*len(train_dataloader))) == 0:

                    accelerator.print(f"***** Evaluating loss,  {global_step}/{cfgs.train.num_train_epochs * len(train_dataloader)} *****")
                    Metrics = evaluate_model_for_math(model, eval_dataloader,accelerator.device)
                    eval_stats = {
                        'eval/loss':Metrics[cfgs.dataset.data_type]['loss'] ,
                        'eval/acc': round(Metrics[cfgs.dataset.data_type]['acc']*100,2) ,
                        'eval/F1': round(Metrics[cfgs.dataset.data_type]['f1']*100,2) ,
                        'eval/precision': round(Metrics[cfgs.dataset.data_type]['precision']*100,2) ,
                        'eval/recall': round(Metrics[cfgs.dataset.data_type]['recall']*100,2) ,
                    }
                    accelerator.print(f"step:{global_step},  losses:{Metrics[cfgs.dataset.data_type]['loss']}, acc: {round(Metrics[cfgs.dataset.data_type]['acc']*100,2)} , F1 score {Metrics[cfgs.dataset.data_type]['f1']}")
                    if accelerator.is_main_process:
                        wandb_log(eval_stats, global_step)

                        if Metrics[cfgs.dataset.data_type]['f1'] > best_f1:
                            best_f1 = Metrics[cfgs.dataset.data_type]['f1']
                            unwrapped_model = accelerator.unwrap_model(model)
                            accelerator.print('save_model')
                            torch.save(unwrapped_model.state_dict(), os.path.join(file_path,f'epoch{epoch}-eval-f1-{best_f1:.2f}.pth'))
                            accelerator.print('save_model_finshed')
                            remove_file(file_path)
                    model.train()

def remove_file(save_dir,k=1):
    model_files = glob(os.path.join(save_dir, '*.pth'))
    model_files.sort(key=lambda x: float(x.split('-f1-')[1].split('.pth')[0]) if 'loss' in x else float('-inf'),reverse=True)
    if len(model_files) > k:
        files_to_delete = model_files[k:]
        for file_to_delete in files_to_delete:
            os.remove(file_to_delete)

if __name__ == "__main__":
    main()