import json
import argparse
import sys
sys.path.append('/workspace/ye')
from utils import *
import os
from vllm import LLM, SamplingParams
from transformers import LlamaTokenizer
import torch
from datasets import load_dataset
def save_jsonL(save_path, result_list):
    parent_dir = os.path.dirname(save_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    with open(save_path, 'w',encoding='utf-8') as f:
        for item in result_list:
            json.dump(item, f,ensure_ascii=False)
            f.write('\n')

def save_json_(save_path, json_obj):
    parent_dir = os.path.dirname(save_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    with open(save_path, 'w',encoding='utf-8') as f:
        json.dump(json_obj, f, indent=4,ensure_ascii=False)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--top_k", type=int, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1, help="")
    parser.add_argument(
                        "--dtype",
                        type=str,
                        help="float16 or int8",
                        choices=["int8", "float16"],
                        default="float16",
                        )
    parser.add_argument("--presence_penalty", type=float, default=-1)
    parser.add_argument("--frequency_penalty", type=float, default=-1)
    parser.add_argument("--data_path", type=str, help="", required=True)
    parser.add_argument("--data_type", type=str, help="", default="sharegpt")
    parser.add_argument("--max_gen_length", type=int, help="", default=1024)
    parser.add_argument("--num_return_sequences", type=int, default=1, help="")
    parser.add_argument("--num_samples", type=int, default=None, help="")
    parser.add_argument("--checkpoint_model_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=8)

    parser.add_argument("--save_path", type=str, default=None, help="")
    parser.add_argument("--style", type=str, default="", help="")

    return parser.parse_args()


class LLMInference:
    def __init__(self, args):
        self.args = args
        self.dataset = load_dataset_(args.data_path, args.data_type)
        if self.args.num_samples is not None:
            self.dataset = self.dataset[:self.args.num_samples]
        self.style = args.style
        tokenizer = AutoTokenizer.from_pretrained('/workspace/ye/llm/mistral-7b', trust_remote_code=True)
        tokenizer.add_special_tokens({'additional_special_tokens':["<step>"]})
        self.model = LLM(model=args.checkpoint_model_path,tensor_parallel_size=args.num_gpus,trust_remote_code=True)
        self.model.set_tokenizer = tokenizer
        self.gen_kwargs = {
            "top_p": args.top_p,
            'seed':1,
            # "top_k": args.top_k,
            # "temperature": args.temperature,
            "max_tokens": args.max_gen_length,
            "n": args.num_return_sequences,
            # "presence_penalty": args.presence_penalty,
            # "frequency_penalty": args.frequency_penalty,
            "skip_special_tokens": False,
            "stop_token_ids": [32000],
            'stop':['<step>']
        }
        self.sampling_params = SamplingParams(**self.gen_kwargs)


    def generate(self, batch):
        output_text = []
        outputs = self.model.generate(batch, self.sampling_params)
        for output in outputs:
            for i in range(self.args.num_return_sequences):
                output_text.append(output.outputs[i].text)
        return output_text

    def combine_prompt(self, sample):
        if self.style == "sharegpt":
            input_ = sample["prompt"]
        elif self.style == 'hh-rlhf':
            pass
        elif self.style == 'oasst1':
            input_ = sample["instruction"]
        elif self.style == 'Gsm8K':
            input_ = "Question:"+sample["question"]+'\n\Answer:'
        elif self.style == 'Math':
            input_ = "Question:"+sample["problem"]+'\n\nAnaswer:'
        else:
            raise ValueError(f"{self.style} not recognized.")
        return input_

    def process(self):
        data_list = self.dataset
        # result_list = []
        sub_inputs = list(map(self.combine_prompt, data_list))
        output_str_list = self.generate(sub_inputs)
        if self.style == 'gov_data' or self.style == 'gov_retrive_sft':
            save_data = []
            for idx in range(len(data_list)):
                save_data.append({
                    'question':data_list[idx]['question'],
                    'groud_truth':data_list[idx]['response'],
                    'response':output_str_list[
                            idx
                            * self.args.num_return_sequences : (idx + 1)
                            * self.args.num_return_sequences
                        ]
                })
            return save_data
        else:
            for idx in range(len(data_list)):
                data_list[idx].update(
                    {
                        "response": output_str_list[
                            idx
                            * self.args.num_return_sequences : (idx + 1)
                            * self.args.num_return_sequences
                        ]
                    }
                )

        return data_list


def load_dataset_(data_path, data_type):
    if data_type == "sharegpt":
        with open(data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    elif data_type == "oasst1":
        data = load_json(data_path)
    elif data_type == "Gsm8K":
        data = load_json(data_path)
    elif data_type == "Math":
        data = load_json(data_path)
    else:
        raise ValueError("{} not recognized.".format(data_type))

    return data


if __name__ == "__main__":
    args = get_args()

    engine = LLMInference(args)

    print(
        f"*** Starting to generate {args.max_gen_length}, prompts={len(engine.dataset)}",
    )
    result_list = engine.process()
    # print(args.save_path)
    # print(result_list)
    with open(args.save_path, 'w' , encoding='utf-8') as f:
        json.dump(result_list, f, indent=4 , ensure_ascii=False)
    # save_json_(args.save_path, result_list)
