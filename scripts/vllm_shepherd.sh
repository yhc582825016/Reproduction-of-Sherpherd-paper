export PYTHONPATH=$PYTHONPATH:/workspace/ye/SFT
NUM_GPUS=4
DATE_TODAY=314
checkpoint_model_path='./checkpoint-530'
num_return_sequences=3
DATA_TYPE=Gsm8k
TEST_DATA_PATH="./dataset/Gsm8k/test.jsonl"
#/workspace/ye/dataset/Math/test.jsonl
#/workspace/ye/dataset/prm800k-main/prm800k/math_splits/test.jsonl
SAVE_PATH="./infer_result/${DATE_TODAY}/"
NUM_SAMPLES=12000
mkdir -p $SAVE_PATH
mkdir -p $SAVE_PATH/finished_data
START_TIME=$(date +%s)
CUDA_VISIBLE_DEVICES=3,4,5,7 python vllm_shepherd_gsm8k.py \
    --num_gpus $NUM_GPUS \
    --temperature 1 \
    --data_path $TEST_DATA_PATH \
    --checkpoint_model_path $checkpoint_model_path \
    --max_gen_length 1024 \
    --num_return_sequences $num_return_sequences \
    --data_type $DATA_TYPE \
    --num_samples $NUM_SAMPLES \
    --style $DATA_TYPE \
    --date_today $DATE_TODAY
END_TIME=$(date +%s)
ELAPSED_TIME=$(($END_TIME - $START_TIME))
echo "Total computation time: $ELAPSED_TIME seconds"
