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
export PYTHONPATH=$PYTHONPATH:./shepherd
# deepspeed --include localhost:1,2
export WANDB_API_KEY=04b01529fb630482bdf2f363456479f197ac5694
OUTPUT=./shepherd/RM/log
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 accelerate launch --config_file ./shepherd/RM/yaml/config.yaml  --main_process_port 29504\
                    ./shepherd/RM/PRM.py \
                    --config_file ./shepherd/RM/yaml/rm.yaml
                    # &> $OUTPUT/training.log
# nohup bash ./shepherd/AI_FeedBack/RM/run.sh > ./shepherd/AI_FeedBack/RM/train.log > 2>&1 &