import argparse
import os
import math
import sys
from shutil import copyfile,rmtree
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
    LlamaTokenizer,
    LlamaForCausalLM,
    MistralForCausalLM,
)
from torch.optim.lr_scheduler import OneCycleLR
from config import sft_config as cfgs, update_config
from accelerate import Accelerator
import torch.nn as nn
from tqdm import tqdm
from peft import LoraConfig,get_peft_model,TaskType,AdaLoraConfig
from Load_data import create_sft_dataloder,SFTDataset
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

def main():
    args = parse_args()
    update_config(cfgs, args)
    mkdir(cfgs.log.output_dir)
    cfgs.seed = 42

    if accelerator.is_local_main_process:
        init_tracker(cfgs)
        cfg_save_path = os.path.join(cfgs.log.output_dir, os.path.basename(args.config_file))
        copyfile(args.config_file, cfg_save_path)

    set_random_seed(cfgs.seed)
    config = AutoConfig.from_pretrained(cfgs.model.model_path,trust_remote_code=True)
    if cfgs.model.model_type == 'llama':
        tokenizer = LlamaTokenizer.from_pretrained(cfgs.model.model_path)
        tokenizer.pad_token_id=2
        tokenizer.eos_token_id=2
        model = LlamaForCausalLM.from_pretrained(cfgs.model.model_path,config=config)
    elif cfgs.model.model_type == 'chatglm':
        tokenizer = AutoTokenizer.from_pretrained(cfgs.model.model_path,trust_remote_code=True)
        model = AutoModel.from_pretrained(cfgs.model.model_path,trust_remote_code=True,config=config)
    elif cfgs.model.model_type == 'opt':
        tokenizer = AutoTokenizer.from_pretrained(cfgs.model.model_path,trust_remote_code=True)
        model = AutoModel.from_pretrained(cfgs.model.model_path,trust_remote_code=True,config=config)
    elif cfgs.model.model_type == 'mistral':
        tokenizer = AutoTokenizer.from_pretrained(cfgs.model.model_path,trust_remote_code=True)
        model = MistralForCausalLM.from_pretrained(cfgs.model.model_path,trust_remote_code=True,config=config)
        tokenizer.pad_token_id=0
        tokenizer.eos_token_id=2
        tokenizer.add_special_tokens({'additional_special_tokens':["<step>"]})
        model.resize_token_embeddings(len(tokenizer))
    if cfgs.train.lora_version=='lora':
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.is_parallelizable = True
        model.model_parallel = True
        model.config.use_cache = (
            False  # silence the warnings. Please re-enable for inference!
        )

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=cfgs.train.lora_rank,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query_key_value"]
        )
        model = get_peft_model(model, peft_config)
        for name, param in model.named_parameters():
            if 'lora' in name:
                with torch.no_grad():  
                    param.data = param.data.float()
    elif cfgs.train.lora_version=='Adalora':
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.is_parallelizable = True
        model.model_parallel = True
        model.config.use_cache = (
            False  # silence the warnings. Please re-enable for inference!
        )

        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=cfgs.train.lora_rank,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query_key_value"]
        )
        model = get_peft_model(model, peft_config)
        for name, param in model.named_parameters():
            if 'lora' in name:
                with torch.no_grad():  
                    param.data = param.data.float()

    optimizer = torch.optim.AdamW(model.parameters(), cfgs.train.learning_rate)
    train_dataset = SFTDataset(config=cfgs,tokenizer=tokenizer,mode='train')
    val_dataset = SFTDataset(config=cfgs,tokenizer=tokenizer,mode='eval')
    train_dataloader, eval_dataloader = create_sft_dataloder(cfgs, train_dataset, val_dataset)

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
    best_loss= evaluate_model(model, eval_dataloader,accelerator)
    accelerator.print(f"step:{global_step}, loss:{best_loss}")
    eval_stats = {
        'eval/loss':best_loss ,
    }
    file_path = os.path.join(cfgs.train.save_path,cfgs.model.model_type,cfgs.log.run_name)
    if accelerator.is_local_main_process:
        wandb_log(eval_stats, global_step)
        
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)
    accelerator.print("***** Running training *****")

    for epoch in range(cfgs.train.num_train_epochs):
        accelerator.print(f"Beginning of Epoch {epoch+1}/{cfgs.train.num_train_epochs}, Total Micro Batches {len(train_dataloader)}")
        model.train()
        with tqdm(enumerate(train_dataloader),total = len(train_dataloader),desc=f"current epoch : {epoch}",\
                disable=(not accelerator.is_local_main_process)) as pbar:
            for step ,batch in pbar:
                batch = to_device(batch, accelerator.device)
                loss = model(**batch,return_dict=True).loss
                accelerator.backward(loss)
                pbar.set_description("loss:%.2f" % (loss))
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                stats = {'train/loss': loss.item(),
                        'lr': optimizer.param_groups[0]['lr']}
                global_step += 1
                if accelerator.is_local_main_process:
                    wandb_log(stats, global_step)
                if global_step % int(len(train_dataloader)) == 0:
                    # Evaluate perplexity on the validation set.
                    accelerator.print(f"***** Evaluating loss,  {global_step}/{cfgs.train.num_train_epochs * len(train_dataloader)} *****")
                    eval_loss = evaluate_model(model, eval_dataloader,accelerator)
                    eval_stats = {
                        'eval/loss':eval_loss,
                    }
                    accelerator.print(f"step:{global_step}, losses:{eval_loss}")
                    # if accelerator.is_local_main_process:
                    #     wandb_log(eval_stats, global_step)
                #         if eval_loss < best_loss:
                #             best_loss = eval_loss
                #             # unwrapped_model = accelerator.unwrap_model(model)
                #             accelerator.print('save_model')
                #             if not cfgs.train.lora_version:
                #                 accelerator.save_model(model, os.path.join(file_path,f'epoch{epoch}-eval-loss-{best_loss:.2f}'))
                #                 # remove_file(file_path)
                #             else:
                #                 model.save_pretrained(file_path)
                #     accelerator.print('save_model_finshed')   
                    # model.train()
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.print('save_model')
        accelerator.save_model(unwrapped_model, os.path.join(file_path,f'epoch{epoch}-eval-loss-{eval_loss:.2f}'))
        accelerator.print('save_model_finshed')

def remove_file(save_dir,k=3):
    model_files = glob(os.path.join(save_dir, '*'))
    model_files.sort(key=lambda x: float(x.split('-loss-')[1]) if 'loss' in x else float('inf'))
    if len(model_files) > k:
        files_to_delete = model_files[k:]
        for file_to_delete in files_to_delete:
            if os.path.isdir(file_to_delete):
                rmtree(file_to_delete)  # 如果是目录，则递归删除
            elif os.path.isfile(file_to_delete):
                os.remove(file_to_delete)  # 如果是文件，则直接删除
            else:
                print(f"{file_to_delete} does not exist or is not a file/directory.")

if __name__ == "__main__":
    main()