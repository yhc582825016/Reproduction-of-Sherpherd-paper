export PYTHONPATH=$PYTHONPATH:./shepherd
# deepspeed --include localhost:1,2
CUDA_VISIBLE_DEVICES=0,1,2,3,4 accelerate launch --config_file ./shepherd/RM/yaml/config.yaml \
                    ./shepherd/RM/PRM_infer.py \
                    --config_file ./shepherd/RM/yaml/rm.yaml
# nohup bash ./shepherd/AI_FeedBack/RM/run.sh > ./shepherd/AI_FeedBack/RM/train.log > 2>&1 &