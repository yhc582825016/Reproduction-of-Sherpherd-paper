# pip install deepspeed
# pip install transformers
# pip install sentencepiece
# pip install datasets
# pip install accelerate
# pip install wandb
# pip install loguru
# pip install yacs
# pip install nvitop
# pip install jsonlines
# pip install peft
# pip install trl
# pip install bitsandbytes
export PYTHONPATH=$PYTHONPATH:./SFT
export WANDB_API_KEY=04b01529fb630482bdf2f363456479f197ac5694

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 accelerate launch --config_file ./SFT/ymal/conf.yaml --main_process_port 29502\
                    ./SFT/main.py \
                --config_file ./SFT/ymal/sft.yaml
# accelerate launch \
#      --config_file ./SFT/ymal/conf.yaml \
#      --machine_rank $MLP_ROLE_INDEX --main_process_ip $MLP_WORKER_0_HOST --main_process_port $MLP_WORKER_0_PORT \
#      ./SFT/main.py \
#      --config_file ./SFT/ymal/sft.yaml