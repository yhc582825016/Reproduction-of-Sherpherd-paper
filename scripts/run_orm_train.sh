export PYTHONPATH=$PYTHONPATH:./shepherd
# deepspeed --include localhost:1,2
export WANDB_API_KEY=04b01529fb630482bdf2f363456479f197ac5694
# pip install yacs
OUTPUT=./shepherd/RM/log
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file ./shepherd/RM/yaml/config.yaml  --main_process_port 29502\
                    ./shepherd/RM/ORM.py \
                    --config_file ./shepherd/RM/yaml/rm.yaml
                    &> $OUTPUT/training.log
# nohup bash ./shepherd/AI_FeedBack/RM/run.sh > ./shepherd/AI_FeedBack/RM/train.log > 2>&1 &