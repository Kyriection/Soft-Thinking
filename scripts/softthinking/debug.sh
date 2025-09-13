model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
save_dir="results/amc23/r1-1.5b_softthinking"
GPU=0

CUDA_VISIBLE_DEVICES=${GPU} python -u eval_MATH_softthinking.py \
    --model_name_or_path $model \
    --save_dir $save_dir \
    --max_tokens 32768 \
    --use_chat_format \
    --dataset "math500" \
    --remove_bos

CUDA_VISIBLE_DEVICES=${GPU} python -u read.py \
    --save_dir $save_dir \
    --model $model