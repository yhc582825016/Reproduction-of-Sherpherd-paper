export PYTHONPATH=$PYTHONPATH:./shepherd
# deepspeed --include localhost:1,2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 accelerate launch --config_file ./shepherd/RM/yaml/config.yaml \
                    ./shepherd/RM/ORM_infer.py \
                    --config_file ./shepherd/RM/yaml/rm.yaml