from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, default_data_collator
from dataclasses import dataclass
import json
from typing import List
import jsonlines
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer, BertTokenizer
from loguru import logger
from typing import List, Dict, Any

def create_sft_dataloder(cfgs,train_dataset,eval_dataset):
    train_sampler = RandomSampler(train_dataset)
    eval_sampler = SequentialSampler(eval_dataset)

    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=default_data_collator,
                                  sampler=train_sampler,
                                  batch_size=cfgs.train.per_device_train_batch_size,
                                  pin_memory=True)
                                  
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=default_data_collator,
                                 sampler=eval_sampler,
                                 batch_size=cfgs.train.per_device_eval_batch_size,
                                 pin_memory=True)
    return train_dataloader, eval_dataloader

def load_json(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except:
        with open(json_path, 'r') as f:
            data = f.readlines()
        data = [json.loads(d) for d in data]
    return data

class SFTDataset(Dataset):
    def __init__(self, config,tokenizer, mode):
        self.config = config
        self.tokenizer = tokenizer
        data_path = {
            "train": self.config.dataset.train_data_path,
            "eval": self.config.dataset.val_data_path,
            "test": self.config.dataset.test_data_path,
        }
        self.data = load_json(data_path[mode])
        self.max_source_length = config.dataset.max_source_length
        self.max_target_length = config.dataset.max_target_length
        self.max_seq_length = config.dataset.max_source_length + config.dataset.max_target_length + 1
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_item = self.data[index]
        if self.config.dataset.data_type=='math':
            question = 'Question:'+data_item['problem']
            answer = '\n\nAnswer:'+data_item['solution']
        else:
            question = 'Question:'+data_item['question']
            answer = '\n\nAnswer:'+data_item['answer']

        a_ids = self.tokenizer.encode(text=question, add_special_tokens=True, truncation=True,
                                      max_length=self.max_source_length)
        b_ids = self.tokenizer.encode(text=answer, add_special_tokens=False, truncation=True,
                                      max_length=self.max_target_length)

        context_length = len(a_ids)
        input_ids = a_ids + b_ids + [self.tokenizer.eos_token_id]
        labels = [self.tokenizer.pad_token_id] * context_length + b_ids + [self.tokenizer.eos_token_id]

        pad_len = self.max_seq_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels = labels + [self.tokenizer.pad_token_id] * pad_len
        labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]

        assert len(input_ids) == len(labels), f"length mismatch: {len(input_ids)} vs {len(labels)}"

        return {
            "input_ids": input_ids,
            "labels": labels
        }
