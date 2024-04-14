GPUS=$1
# The name of experiment
NAME=$2

# Create dirs and make backup
output=/data/bert_output/$NAME
echo $output
mkdir $output
mkdir -p $output/src

export TRAIN_FILE=/home/ytw/dataset/wiki/en.train.raw
export TEST_FILE=/home/ytw/dataset/wiki/en.valid.raw

# Pre-training
CUDA_VISIBLE_DEVICES=$GPUS python3 /home/ytw/code/LEMON/lm/vlm/run_lm_distributed.py \
    --output_dir=$output \
	--overwrite_output_dir \
	--tokenizer_name=bert-base-uncased \
    --cache_dir=/home/ytw/dataset/wiki/cache_dir \
    --model_type=modbert \
	--block_size=126 \
	--per_gpu_train_batch_size=64 \
    --per_gpu_eval_batch_size=64 \
	--gradient_accumulation_steps=1 \
	--weight_decay=0.01 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --col_data \
    --split_sent \
    --eval-when-log \
    --mlm ${@:3} | tee $output/log.log

    #--fp16 \
	#--fp16_opt_level O2 \
