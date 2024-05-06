import json
import argparse
import sys
sys.path.append('/workspace/ye')
from utils import *
import os
from vllm import LLM, SamplingParams
import random
import time
import re
random.seed(time.time())
seed = random.randint(0,10000)
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
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=0, help="")
    parser.add_argument(
                        "--dtype",
                        type=str,
                        help="float16 or int8",
                        choices=["int8", "float16"],
                        default="float16",
                        )
    parser.add_argument("--base_model", type=str, default=-1)
    parser.add_argument("--presence_penalty", type=float, default=-1)
    parser.add_argument("--frequency_penalty", type=float, default=-1)
    parser.add_argument("--data_path", type=str, help="", required=True)
    parser.add_argument("--data_type", type=str, help="", default="sharegpt")
    parser.add_argument("--max_gen_length", type=int, help="", default=1024)
    parser.add_argument("--num_return_sequences", type=int, default=1, help="")
    parser.add_argument("--date_today", type=int, default=1, help="")
    parser.add_argument("--num_samples", type=int, default=None, help="")
    parser.add_argument("--checkpoint_model_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--iter", type=int, default=None)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=None, help="")
    parser.add_argument("--style", type=str, default="", help="")

    return parser.parse_args()

class LLMInference:
    def __init__(self, args):
        self.args = args
        self.style = args.style
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        tokenizer.add_special_tokens({'additional_special_tokens':["<step>"]})
        self.model = LLM(model=args.checkpoint_model_path,tensor_parallel_size=args.num_gpus,trust_remote_code=True)
        self.model.set_tokenizer = tokenizer
        self.gen_kwargs = {
            # "top_p": args.top_p,
            # "top_k": args.top_k,
            "temperature": args.temperature,
            "max_tokens": args.max_gen_length,
            "n": args.num_return_sequences,
            'seed':seed,
            # "presence_penalty": args.presence_penalty,
            # "frequency_penalty": args.frequency_penalty,
            "skip_special_tokens": False,
            # "stop_token_ids": [32000,2],
            # 'stop':['<step>',"</s>"]
        }
        self.sampling_params = SamplingParams(**self.gen_kwargs)

    def generate(self, batch):
        output_text = []
        outputs = self.model.generate(batch, self.sampling_params)
        for output in outputs:
            for i in range(self.args.num_return_sequences):
                output_text.append(output.outputs[i].text)
        return output_text

def combine_prompt(sample):
    if args.style == 'Gsm8k':
        input_ = "Question:"+sample["question"]+'\n\nAnswer:'
    elif args.style == 'Math':
        input_ = "Question:"+sample["problem"]+'\n\nAnaswer:'
    return input_

def load_dataset_(data_path, data_type):
    if data_type == "Gsm8k":
        data = load_json(data_path)
    elif data_type == "Math":
        data = load_json(data_path)
    else:
        raise ValueError("{} not recognized.".format(data_type))
    return data

def process_step(data, is_initial=False):
    """
    Processes each step in the dataset, extracting and formatting the required information.
    Handles both initial data and updated data after generating new responses.
    """
    processed_data = []
    for item in data:
        if is_initial:
            question = item['problem']  # Assuming initial data has 'response' key.
            responses = item['solution']
        else:
            question = item['problem'][0]  # Updated data uses 'responses'.
            responses = item['solution'][0]
        random_idx = random.randint(0, len(responses)-1) if responses else 0
        answer = item['answer']
        return_result = []
        for res in responses:
            match_result = re.findall(r'\\boxed\{(.*)\}',res)
            if match_result !=[]:
                match_result=match_result[0]
            else:
                match_result = None
            return_result.append(match_result)

        step_label = any(element == answer for element in return_result)
        # print('response',responses[random_idx])
        response = responses[random_idx].split("<step>")[0] if responses else ''
        # print('response',response)
        if 'step_labels' in item:
            step_labels = item['step_labels']
            step_labels.append(int(step_label))
        else:
            step_labels = [int(step_label)]
        if is_initial:
            formatted_question = f"Question:{question}\n\nAnswer:{response}<step>"
        else:
            formatted_question = f"{question}{response}<step>"
        assert len(step_labels) == formatted_question.count("<step>")
        temp = {
            'problem': [formatted_question],
            'answer': answer,
            'step_labels': step_labels
        }
        processed_data.append(temp)
    return processed_data

def update_step_data(data, output_str_list, args):
    """
    Updates the data with new responses generated by the engine, handling step labels.
    """
    updated_data = []
    for i, item in enumerate(data):
        outputs_per_question = len(item['problem'])*args.num_return_sequences
        start_idx = i * outputs_per_question
        end_idx = start_idx + outputs_per_question
        original_list = output_str_list[start_idx:end_idx]
        item["solution"] = [original_list[j:j+3] for j in range(0, len(original_list), 3)]
        # Further processing based on new responses
        processed = process_step([item])
        updated_data.extend(processed)
    return updated_data

if __name__ == "__main__":
    args = get_args()
    engine = LLMInference(args)
    engine.sampling_params.temperature  = random.uniform(0.9, 1.1)

    for iter in range(1):
        dataset = load_dataset_(args.data_path, args.data_type)
        if args.num_samples is not None:
            dataset = dataset[:args.num_samples]
        sub_inputs = list(map(combine_prompt, dataset))
        output_str_list = engine.generate(sub_inputs)
        for idx in range(len(dataset)):
            dataset[idx].update(
                {
                    "solution": output_str_list[
                        idx
                        * args.num_return_sequences : (idx + 1)
                        * args.num_return_sequences
                    ]
                }
            )
        print(f"*** Starting to generate {args.max_gen_length}, prompts={len(dataset)}")
        finished_data = []
        save_data = process_step(dataset, is_initial=True)
        print('save_data',save_data)
        save_data_ = []
        for item in save_data:
            if "boxed{" in item['problem'][0] or '</s>' in item['problem'][0]:
                finished_data.append(item)
            else:
                save_data_.append(item)
        save_data = save_data_
        loop_iter = 0
        while save_data:
            print(f'Current len(save_data):{len(save_data)}')
            sub_inputs = [item['problem'][0] for item in save_data]
            output_str_list = engine.generate(sub_inputs)
            save_data = update_step_data(save_data, output_str_list, args)
            save_data_ = []
            for item in save_data:
                if "boxed{" in item['problem'][0] or '</s>' in item['problem'][0]:
                    finished_data.append(item)
                else:
                    save_data_.append(item)
            save_data = save_data_
            loop_iter+=1
            if loop_iter==30:
                break

        with open(f'/workspace/ye/SFT/infer_result/{args.date_today}/finished_data/math-infer-{iter}.jsonl', "w", encoding='utf-8') as f:
            json.dump(finished_data, f, ensure_ascii=False, indent=4)
        with open(f'/workspace/ye/SFT/infer_result/{args.date_today}/finished_data/math-infer-{iter}-unprocessed.jsonl', "w", encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=4)