# %%
import torch
from torch._C import NoopLogger
import torch.nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
import torch.distributed as dist
import os
import json
import yaml
import wandb
from transformers import BertModel, BertPreTrainedModel,AutoConfig
from transformers import set_seed, AutoTokenizer
# from transformers import RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutput, Seq2SeqLMOutput
from torch.utils.data import Dataset
import random 
import numpy as np
from glob import glob
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from dataclasses import dataclass
from typing import List, Dict, Any
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import deepspeed
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score,classification_report

class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config):
        super().__init__()
        # Use a two-layer MLP to encode the prefix
        self.embedding = torch.nn.Embedding(config.pre_seq_len, config.hidden_size)
        self.trans = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
        )


    def forward(self, prefix: torch.Tensor):
        prefix_tokens = self.embedding(prefix)
        past_key_values = self.trans(prefix_tokens)
        return past_key_values

# %%
class BertPrefixForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        bert_param = 0
        for name, param in self.bert.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('all_param  is {}'.format(all_param)) # 9860105
        print('freeze param is {}'.format(bert_param))
        print('trainable param is {}'.format(total_param))
    
    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2) # layer bs head pre_seq_len n_embd
        return past_key_values

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        # past_key_values = None
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        # logits = F.softmax(logits,dim=-1)
        return outputs
   
def get_model(model_args, config: AutoConfig, fix_bert: bool = False):
    config.hidden_dropout_prob = model_args.model.hidden_dropout_prob
    config.pre_seq_len = model_args.model.pre_seq_len
    config.prefix_hidden_size = model_args.model.prefix_hidden_size
    
    model_class = BertPrefixForSequenceClassification
    model = model_class.from_pretrained(
        model_args.model.model_path,
        config=config )

    return model


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def load_json(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except:
        with open(json_path, 'r') as f:
            data = f.readlines()
        data = [json.loads(d) for d in data]
    return data
    
def save_json(save_path, data):
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

def load_jsonL(json_path):

    with open(json_path, 'r') as f:
        data = f.readlines()

    data = [json.loads(d) for d in data]

    return data

def save_jsonL(save_path, data):
    with open(save_path, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

def is_local_main_process(rank_idx):
    return rank_idx <= 0

def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)

def gather_objects(objs):
    output_list = []
    world_size = dist.get_world_size()
    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, objs)
    for i in range(world_size):
        output_list.extend(output[i])
    # 将对象转换回字符串列表
    return output_list


def init_tracker(cfgs):
    wandb.init(
        dir=cfgs.log.output_dir,
        project=cfgs.log.project_name,
        name=cfgs.log.run_name,
        entity=None,
        group=None,
        mode="disable" if os.environ.get("debug", False) else "online"
    )
    # 更新配置参数
    wandb.config.update(cfgs)


def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_optimizer_grouped_parameters(model,
                                     weight_decay,
                                     no_decay_name_list=[
                                         "bias", "LayerNorm.weight"
                                     ]):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n
                            for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
    ]
    return optimizer_grouped_parameters

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                json_obj = json.loads(line.strip())
                data.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"JSON 解析错误: {e}")
    
    return data
import json

def save_json(filename,data):
    """
    保存数据到JSON文件。

    :param data: 要保存的数据，应该是可以被json模块处理的数据类型。
    :param filename: 保存文件的名称，应该包括.json后缀。
    """
    try:
        # 打开文件用于写入
        with open(filename, 'w', encoding='utf-8') as file:
            # 将数据转换为JSON格式并写入文件
            json.dump(data, file, ensure_ascii=False, indent=4)
        print(f"数据已成功保存到 {filename}")
    except Exception as e:
        # 如果出现错误，打印错误信息
        print(f"保存数据时出错: {e}")

    
def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        output[k] = v.to(device)
    return output


def log_stats(stats, global_rank, steps):
    if is_local_main_process(global_rank):
        for k, v in stats.items():
            if isinstance(v, dict):
                for k_, v_ in stats[k].items():
                    wandb.log({f'{k}/{k_}': v_}, step=steps)
            else:
                wandb.log({k: v}, step=steps)  

def wandb_log(stats,steps):
    for k, v in stats.items():
        if isinstance(v, dict):
            for k_, v_ in stats[k].items():
                wandb.log({f'{k}/{k_}': v_}, step=steps)
        else:
            wandb.log({k: v}, step=steps)  


def save_rm_model(cfgs, tokenizer, model, subfolder=None):

    if cfgs.global_rank == 0:
        save_hf_format(model, tokenizer, cfgs, sub_folder=subfolder)

    if cfgs.deepspeed.zero_stage == 3:
        # For zero stage 3, each gpu only has a part of the model, so we need a special save function
        save_path = os.path.join(cfgs.log.output_dir, subfolder)
        save_zero_three_model(model,
                                cfgs.global_rank,
                                save_path,
                                zero_stage=cfgs.deepspeed.zero_stage)
        
def save_hf_format(model, tokenizer, cfgs, sub_folder=""):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = os.path.join(cfgs.log.output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir) # 130B会报错，暂时注释掉，待确认下问题



def save_zero_three_model(model_ema, global_rank, save_dir, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    os.makedirs(save_dir, exist_ok=True)
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)

    model_to_save = model_ema.module if hasattr(model_ema,
                                                'module') else model_ema
    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():

            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v
                                                                            ]),
                                                       enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
        del output_state_dict

def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]

def gather_objects(objs):
    """
    """
    output_list = []
    world_size = dist.get_world_size()
    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, objs)
    for i in range(world_size):
        output_list.extend(output[i])
    # 将对象转换回字符串列表
    return output_list

def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor

def evaluate_model(model, eval_dataloader,accelerator):
    model.eval()
    losses = 0
    with tqdm(enumerate(eval_dataloader),total = len(eval_dataloader),desc=f"current evaling",\
            disable=(not accelerator.is_local_main_process)) as pbar:
        for step ,batch in pbar:
            batch = to_device(batch, model.device)
            with torch.no_grad():
                outputs = model(**batch,return_dict=True)
            loss = outputs.loss
            losses += loss.float()
    losses_all = get_all_reduce_mean(losses).item()
    return losses_all