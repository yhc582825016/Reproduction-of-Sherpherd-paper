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
from prompt import *
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score
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
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
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

        pooled_output = outputs[1]
        pooled_output = F.normalize(pooled_output, p=2, dim=-1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # logits = F.softmax(logits,dim=-1)
        return logits
   
def get_model(model_args, config: AutoConfig, fix_bert: bool = False):
    config.hidden_dropout_prob = model_args.hidden_dropout_prob
    config.pre_seq_len = model_args.pre_seq_len
    config.prefix_hidden_size = model_args.prefix_hidden_size
    
    model_class = BertPrefixForSequenceClassification
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        config=config )

    return model


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
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

def save_json(data, filename):
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

class CustomDataset_For_Rankloss(Dataset):
    """
    Args:
        data_dir (str): Data directory
        suffix (str): suffix of data file
    Return:
    """
    def __init__(self, 
                data_dir, 
                tokenizer, 
                max_src_len,
                max_len, 
                model_type,cfgs):
        
        self.data_list = []
        print(data_dir)
        for i in data_dir:
            self.data_list.extend(load_json(i))
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_src_len = max_src_len
        self.model_type = model_type
        self.principles = load_json(cfgs.model.principle_path)
        self.cfgs = cfgs

    def __getitem__(self, idx):
        chosen_input_ids = self.get_ids(self.data_list[idx]['chosen'])
        rejected_input_ids = self.get_ids(self.data_list[idx]['rejected'])
        meta_info = self.data_list[idx]
       
        model_inputs = {
            'chosen_input_ids': chosen_input_ids,
            'rejected_input_ids': rejected_input_ids,
            'meta_info':meta_info
        }
        return model_inputs
    
    def __len__(self):
        return len(self.data_list)
    
    def get_ids(self,data):
        prompt,response = data
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        if self.model_type == 'bert':
            if len(prompt_ids) > self.max_src_len:
                prompt_ids = prompt_ids[-self.max_src_len:]
            if len(prompt_ids) + len(response_ids) +3 > self.max_len:
                response_ids = response_ids[:self.max_len - len(prompt_ids) - 3]
            input_ids = [101]+prompt_ids+[102]+response_ids+[102]
        elif self.model_type in  ['chatglm','gpt2','custom']:
            if len(prompt_ids) > self.max_src_len:
                prompt_ids = prompt_ids[-self.max_src_len:]
            if len(prompt_ids) + len(response_ids) +3 > self.max_len:
                response_ids = response_ids[:self.max_len - len(prompt_ids) - 3]
            input_ids = self.tokenizer.build_inputs_with_special_tokens(prompt_ids,response_ids)
        return input_ids

class CustomDataset_For_Verifiers(Dataset):
    """
    Args:
        data_dir (str): Data directory
        suffix (str): suffix of data file
    Return:
    """
    def __init__(self, 
                data_dir, 
                tokenizer, 
                max_src_len,
                max_len, 
                model_type,cfgs):
        
        self.data_list = []
        self.input_ids = []
        self.labels = []
        self.meta_infos = []
        print(data_dir)
        for i in data_dir:
            self.data_list.extend(load_json(i))
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_src_len = max_src_len
        self.model_type = model_type
        print(f'Total {len(self.data_list)} Samples')
        optimized_input_ids = []
        optimized_labels = []
        optimized_meta_infos = []
        for i, item in enumerate(self.data_list):
            question = item['question']
            responses = item['response']
            labels = item['labels']
            dataset = item['dataset']
            for response, label in zip(responses, labels):
                input_id = self.get_ids(question, response)
                optimized_input_ids.append(input_id)
                optimized_labels.append(label)
                optimized_meta_infos.append({
                    'prompt': question,
                    'response': response,
                    'label': label,
                    'dataset': dataset
                })
        self.input_ids = optimized_input_ids
        self.labels = optimized_labels
        self.meta_infos = optimized_meta_infos
        
        print(f'Total {len(self.labels)} Labels')
        self.cfgs = cfgs

    def __getitem__(self, idx):
        model_inputs = {
            'input_ids': self.input_ids[idx],
            'meta_info':self.meta_infos[idx],
            'labels':self.labels[idx],
        }
        return model_inputs
    
    def __len__(self):
        return len(self.labels)
    
    def get_ids(self,prompt,response):
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response.replace('<endoftext>',""), add_special_tokens=False)
    
        if len(prompt_ids) > self.max_src_len:
            prompt_ids = prompt_ids[-self.max_src_len:]
        if len(prompt_ids) + len(response_ids) +3 > self.max_len:
            response_ids = response_ids[:self.max_len - len(prompt_ids) - 3]
        input_ids = [1]+prompt_ids+[1]+response_ids+[2]
        return input_ids
    
class CustomDataset_For_Step_Verifiers(Dataset):
    """
    Args:
        data_dir (str): Data directory
        suffix (str): suffix of data file
    Return:
    """
    def __init__(self, 
                data_dir, 
                tokenizer, 
                max_src_len,
                max_len, 
                model_type,cfgs):
        
        self.data_list = []
        self.input_ids = []
        self.labels = []
        self.meta_infos = []
        print(data_dir)
        for i in data_dir:
            self.data_list.extend(load_json(i))
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_src_len = max_src_len
        self.model_type = model_type
        print(f'Total {len(self.data_list)} Samples')
        step_token_id = 32000
        optimized_input_ids = []
        optimized_labels = []
        optimized_meta_infos = []

        for item in tqdm(self.data_list,disable=not cfgs.is_local_main_process,desc='Process data'):
            question = item['question'][0]
            responses = item['response']
            dataset = item['dataset']
            labels_list = item['labels']
            for response, labels in zip(responses, labels_list):
                input_id = self.get_ids(question, response)
                indices = [i for i, x in enumerate(input_id) if x == step_token_id]
                step_labels = labels[:len(indices)]
                labels_full = len(input_id) * [-100]

                for step_label, indice in zip(step_labels, indices):
                    labels_full[indice] = step_label

                optimized_input_ids.append(input_id)
                optimized_labels.append(labels_full)
                optimized_meta_infos.append({'prompt': question, 'response': response, 'step_labels': step_labels, 'dataset': dataset,"answer":item['answer']})
        self.input_ids = optimized_input_ids
        self.labels = optimized_labels
        self.meta_infos = optimized_meta_infos
        print(f'Total {len(self.labels)} Labels')
        self.cfgs = cfgs

    def __getitem__(self, idx):
        model_inputs = {
            'input_ids': self.input_ids[idx],
            'meta_info':self.meta_infos[idx],
            'labels':self.labels[idx],
        }
        return model_inputs
    
    def __len__(self):
        return len(self.labels)
    
    def get_ids(self,prompt,response):
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response.replace('<endoftext>',""), add_special_tokens=False)
    
        if len(prompt_ids) > self.max_src_len:
            prompt_ids = prompt_ids[-self.max_src_len:]
        if len(prompt_ids) + len(response_ids) +3 > self.max_len:
            response_ids = response_ids[:self.max_len - len(prompt_ids) - 3]
        input_ids = [1]+prompt_ids+[1]+response_ids+[2]
        return input_ids

@dataclass
class DataCollatorWithPadding_For_Math:
    tokenizer: Any
    def __call__(self, inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max([len(x['input_ids']) for x in inputs])
        input_ids = [x['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(x['input_ids'])) for x in inputs]
        batch = dict()
        batch['input_ids'] = torch.tensor(input_ids).view(-1, max_len)
        batch['meta_info'] = [x['meta_info'] for x in inputs]
        batch['labels'] =  torch.tensor([x['labels'] for x in inputs]).view(-1)
        return batch

@dataclass
class DataCollatorWithPadding_For_Step_Math:
    tokenizer: Any
    def __call__(self, inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max([len(x['input_ids']) for x in inputs])
        input_ids = [x['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(x['input_ids'])) for x in inputs]
        labels = [x['labels'] + [-100] * (max_len - len(x['labels'])) for x in inputs]
        batch = dict()
        batch['input_ids'] = torch.tensor(input_ids).view(-1, max_len)
        batch['meta_info'] = [x['meta_info'] for x in inputs]
        batch['labels'] =  torch.tensor(labels).view(-1,max_len)
        return batch

@dataclass
class DataCollatorWithPadding:
    tokenizer: Any    
    def __call__(self, inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(max([len(x['chosen_input_ids']) for x in inputs]),max([len(x['rejected_input_ids']) for x in inputs]))
        chosen_input_ids = [x['chosen_input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(x['chosen_input_ids'])) for x in inputs]
        rejected_input_ids = [x['rejected_input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(x['rejected_input_ids'])) for x in inputs]
        batch = dict()
        batch['chosen_input_ids'] = torch.tensor(chosen_input_ids).view(-1, max_len)
        batch['rejected_input_ids'] = torch.tensor(rejected_input_ids).view(-1, max_len)
        batch['meta_info'] = [x['meta_info'] for x in inputs]
        return batch
    
def create_rm_dataloder_for_math(cfgs, train_dataset, eval_dataset, tokenizer):
    train_sampler = RandomSampler(train_dataset)
    eval_sampler = SequentialSampler(eval_dataset)

    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=DataCollatorWithPadding_For_Math(tokenizer),
                                  batch_size=cfgs.train.per_device_train_batch_size,
                                  sampler=train_sampler,
                                  pin_memory=True)
                                  
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=DataCollatorWithPadding_For_Math(tokenizer),
                                 batch_size=cfgs.train.per_device_eval_batch_size,
                                 pin_memory=True,
                                 sampler=eval_sampler,
                                 shuffle=False)
    return train_dataloader, eval_dataloader

def create_rm_dataloder_for_step_math(cfgs, train_dataset, eval_dataset, tokenizer):
    train_sampler = RandomSampler(train_dataset)
    eval_sampler = SequentialSampler(eval_dataset)

    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=DataCollatorWithPadding_For_Step_Math(tokenizer),
                                  batch_size=cfgs.train.per_device_train_batch_size,
                                  sampler=train_sampler,
                                  pin_memory=True)
                                  
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=DataCollatorWithPadding_For_Step_Math(tokenizer),
                                 batch_size=cfgs.train.per_device_eval_batch_size,
                                 pin_memory=True,
                                 sampler=eval_sampler,
                                 shuffle=False)
    return train_dataloader, eval_dataloader


def create_rm_dataloder(cfgs, train_dataset, eval_dataset, tokenizer):
    train_sampler = RandomSampler(train_dataset)
    eval_sampler = SequentialSampler(eval_dataset, shuffle=False)

    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=DataCollatorWithPadding(tokenizer, cfgs.dataset.max_seq_len),
                                  sampler=train_sampler,
                                  batch_size=cfgs.train.per_device_train_batch_size,
                                  pin_memory=True)
                                  
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=DataCollatorWithPadding(tokenizer, cfgs.dataset.max_seq_len),
                                 sampler=eval_sampler,
                                 batch_size=cfgs.train.per_device_eval_batch_size,
                                 pin_memory=True)
    return train_dataloader, eval_dataloader

    
def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        if isinstance(v,torch.Tensor):
            output[k] = v.to(device)
        else:
            output[k] = v
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

def evaluate_model(model, eval_dataloader):
    model.eval()
    losses = 0
    for step, batch in enumerate(eval_dataloader):
        batch = to_device(batch, model.device)
        with torch.no_grad():
            outputs = model(**batch,return_dict=True)
        loss = outputs.loss
        losses += loss.float()

    losses_all = get_all_reduce_mean(losses).item()
    return losses_all

def compute_loss(model,**inputs):
    chosen_input_ids = inputs["chosen_input_ids"]
    rejected_input_ids = inputs["rejected_input_ids"]
    input_ids = torch.cat([chosen_input_ids,rejected_input_ids],dim=0)
    bs = input_ids.size(0)//2
    sentence_logits = model.module.base_model.model(input_ids=input_ids)[0].permute(1, 0, 2).contiguous()
    # sentence_logits = self.sentence_encoder.transformer(input_ids=input_ids)[0]
    # sentence_logits = self.linear(sentence_logits).permute(1, 0, 2).contiguous()   
    chosen_logits = sentence_logits[:bs]
    reject_logits= sentence_logits[bs:]
    loss = 0
    correct_count = 0
    for i in range(bs):
        chosen_end_ids = torch.where(chosen_input_ids[i]==2)[0][0]
        reject_end_ids = torch.where(rejected_input_ids[i]==2)[0][0]
        chosen_logit = chosen_logits[i,chosen_end_ids,:].unsqueeze(-1)
        reject_logit = reject_logits[i,reject_end_ids,:].unsqueeze(-1)
        loss += -F.logsigmoid(chosen_logit - reject_logit-0.1)##bs*seq
        if chosen_logit > reject_logit:
            correct_count +=1
    return loss.mean(),correct_count

def evaluate_model_for_math(model, eval_dataloader,device):
    model.eval()
    all_info = []
    pred_labels = []
    ground_truths = []
    with tqdm(enumerate(eval_dataloader),total = len(eval_dataloader)) as pbar:
        for step,batch in pbar:
            batch = to_device(batch,device)
            with torch.no_grad():
                # loss,chosen_logits,reject_logits = model('eval',**batch,return_dict=True
                loss,meta_info,bs_correct_count,pred_label= model.forward(**batch)
                # print(meta_info)
                for i in range(len(meta_info)):
                    ground_truths.append(meta_info[i]['label'])
                    pred_labels.append(pred_label[i])
                all_info.extend(meta_info)
    ground_truths = gather_objects(ground_truths)
    pred_labels = gather_objects(pred_labels)
    f1 = f1_score(ground_truths,pred_labels)
    recall = recall_score(ground_truths,pred_labels)
    precision = precision_score(ground_truths,pred_labels)
    all_info = gather_objects(all_info)
    datatype_dic={}
    for i in all_info:
        if i['dataset'] not in datatype_dic:
            datatype_dic[i['dataset']]={'loss':[i['loss']],'correctness':[i['correctness']]}
        else:
            datatype_dic[i['dataset']]['loss'].append(i['loss'])
            datatype_dic[i['dataset']]['correctness'].append(i['correctness'])
    Metrics = {}
    for key in list(datatype_dic.keys()):
        correctness = datatype_dic[key]['correctness']
        loss = datatype_dic[key]['loss']
        Metrics[key]={'acc':sum(correctness)/len(correctness),'loss':sum(loss)/len(correctness),"f1":f1,"recall":recall,'precision':precision}
    return Metrics

def evaluate_model_version(model, eval_dataloader,accelerate,cfgs):
    model.eval()
    all_info = []
    with tqdm(enumerate(eval_dataloader),total = len(eval_dataloader)) as pbar:
        for step,batch in pbar:
            batch = to_device(batch, accelerate.device)
            with torch.no_grad():
                # loss,chosen_logits,reject_logits = model('eval',**batch,return_dict=True
                loss,meta_info,bs_correct_count= model.module.forward('eval',**batch,return_dict=True)
                all_info.extend(meta_info)
    all_info = gather_objects(all_info)
    datatype_dic={}
    for i in all_info:
        if i['dataset'] not in datatype_dic:
            datatype_dic[i['dataset']]={'loss':[i['loss']],'correctness':[i['correctness']]}
        else:
            datatype_dic[i['dataset']]['loss'].append(i['loss'])
            datatype_dic[i['dataset']]['correctness'].append(i['correctness'])
    Metrics = {}
    for key in list(datatype_dic.keys()):
        correctness = datatype_dic[key]['correctness']
        loss = datatype_dic[key]['loss']
        Metrics[key]={'acc':sum(correctness)/len(correctness),'loss':sum(loss)/len(correctness)}
    return Metrics

def evaluate_model_for_step_math(model, eval_dataloader,device):
    model.eval()
    all_info = []
    pred_labels = []
    ground_truths = []
    with tqdm(enumerate(eval_dataloader),total = len(eval_dataloader)) as pbar:
        for step,batch in pbar:
            batch = to_device(batch,device)
            with torch.no_grad():
                # loss,chosen_logits,reject_logits = model('eval',**batch,return_dict=True
                loss,meta_info,correct_count,total_count,pred_label= model.forward(**batch)
                # print(meta_info)
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
    return Metrics

def Metrics():
    pass