GPU=$2
if [ "$1" = "7b_aime" ]; then
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    save_dir="results/AIME/r1-7b_softthinking"

    CUDA_VISIBLE_DEVICES=${GPU} python -u eval_MATH_softthinking.py \
        --model_name_or_path $model \
        --save_dir $save_dir \
        --max_tokens 32768 \
        --use_chat_format \
        --dataset "AIME" \
        --remove_bos

    CUDA_VISIBLE_DEVICES=${GPU} python -u read.py \
        --save_dir $save_dir \
        --model $model

elif [ "$1" = "1.5b_aime" ]; then
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    save_dir="results/AIME/r1-1.5b_softthinking"

    CUDA_VISIBLE_DEVICES=${GPU} python -u eval_MATH_softthinking.py \
        --model_name_or_path $model \
        --save_dir $save_dir \
        --max_tokens 32768 \
        --use_chat_format \
        --dataset "AIME" \
        --remove_bos

    CUDA_VISIBLE_DEVICES=${GPU} python -u read.py \
        --save_dir $save_dir \
        --model $model

elif [ "$1" = "7b_aime25" ]; then
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    save_dir="results/aime25/r1-7b_softthinking"

    CUDA_VISIBLE_DEVICES=${GPU} python -u eval_MATH_softthinking.py \
        --model_name_or_path $model \
        --save_dir $save_dir \
        --max_tokens 32768 \
        --use_chat_format \
        --dataset "aime25" \
        --remove_bos

    CUDA_VISIBLE_DEVICES=${GPU} python -u read.py \
        --save_dir $save_dir \
        --model $model

elif [ "$1" = "1.5b_aime25" ]; then
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    save_dir="results/aime25/r1-1.5b_softthinking"

    CUDA_VISIBLE_DEVICES=${GPU} python -u eval_MATH_softthinking.py \
        --model_name_or_path $model \
        --save_dir $save_dir \
        --max_tokens 32768 \
        --use_chat_format \
        --dataset "aime25" \
        --remove_bos

    CUDA_VISIBLE_DEVICES=${GPU} python -u read.py \
        --save_dir $save_dir \
        --model $model

elif [ "$1" = "7b_math500" ]; then
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    save_dir="results/math500/r1-7b_softthinking"

    CUDA_VISIBLE_DEVICES=${GPU} python -u eval_MATH_softthinking.py \
        --model_name_or_path $model \
        --save_dir $save_dir \
        --max_tokens 32768 \
        --use_chat_format \
        --dataset "MATH500" \
        --remove_bos

    CUDA_VISIBLE_DEVICES=${GPU} python -u read.py \
        --save_dir $save_dir \
        --model $model

elif [ "$1" = "1.5b_math500" ]; then
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    save_dir="results/math500/r1-1.5b_softthinking"

    CUDA_VISIBLE_DEVICES=${GPU} python -u eval_MATH_softthinking.py \
        --model_name_or_path $model \
        --save_dir $save_dir \
        --max_tokens 32768 \
        --use_chat_format \
        --dataset "MATH500" \
        --remove_bos

    CUDA_VISIBLE_DEVICES=${GPU} python -u read.py \
        --save_dir $save_dir \
        --model $model

elif [ "$1" = "7b_amc23" ]; then
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    save_dir="results/amc23/r1-7b_softthinking"

    CUDA_VISIBLE_DEVICES=${GPU} python -u eval_MATH_softthinking.py \
        --model_name_or_path $model \
        --save_dir $save_dir \
        --max_tokens 32768 \
        --use_chat_format \
        --dataset "amc23" \
        --remove_bos

    CUDA_VISIBLE_DEVICES=${GPU} python -u read.py \
        --save_dir $save_dir \
        --model $model

elif [ "$1" = "1.5b_amc23" ]; then
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    save_dir="results/amc23/r1-1.5b_softthinking"

    CUDA_VISIBLE_DEVICES=${GPU} python -u eval_MATH_softthinking.py \
        --model_name_or_path $model \
        --save_dir $save_dir \
        --max_tokens 32768 \
        --use_chat_format \
        --dataset "amc23" \
        --remove_bos

    CUDA_VISIBLE_DEVICES=${GPU} python -u read.py \
        --save_dir $save_dir \
        --model $model

else
    echo "Unknown option: $1"
    echo "Usage: {gpu1|gpu2}"
    exit 1
fi

