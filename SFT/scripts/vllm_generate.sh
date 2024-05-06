export PYTHONPATH=$PYTHONPATH:./SFT
NUM_GPUS=4
DATE_TODAY=309
checkpoint_model_path='./SFT/checkpoint/mistral/gsm8k/checkpoint-310'
TEST_DATA_PATH='./SFT/data/shepherd/test-set.json'
SAVE_PATH="./SFT/infer_result/${DATE_TODAY}"
num_return_sequences=3
DATA_TYPE=Gsm8K
TEMPERATRUE=1
NUM_SAMPLES=100
mkdir -p $SAVE_PATH
# greedy search high data
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./SFT/vllm_inference.py \
    --num_gpus $NUM_GPUS \
    --temperature 1 \
    --data_path $TEST_DATA_PATH \
    --checkpoint_model_path $checkpoint_model_path \
    --save_path ${SAVE_PATH}/Gsm8k-310step-test-set-return3-2.jsonl \
    --max_gen_length 1024 \
    --num_return_sequences $num_return_sequences \
    --data_type $DATA_TYPE \
    --num_samples $NUM_SAMPLES \
    --style $DATA_TYPE