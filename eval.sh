TASKS=(
    # tinyBenchmarks
    tinyHellaswag
    tinyArc
    tinyGSM8k
    tinyMMLU
    tinyTruthfulQA
    tinyWinogrande
)

MODELS=(
    # Qwen/Qwen2.5-3B-Instruct
    Qwen/Qwen2.5-7B-Instruct
    # Qwen/Qwen2.5-14B-Instruct
    # meta-llama/Llama-3.2-3B-Instruct
    # meta-llama/Llama-3.1-8B-Instruct
    # google/gemma-2-9b-it
)

port=9904

for model_id in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        # baseline
        model_name=$(echo "$model_id" | cut -d'/' -f2)
        echo "Evaluating model: $model_id with angle none on port $port"
        lm_eval \
            --model local-completions \
            --tasks ${task} \
            --batch_size 1 \
            --model_args model=${model_id},base_url=http://0.0.0.0:${port}/angular_steering/none,num_concurrent=1,max_retries=3,tokenized_requests=False,max_length=4096 \
            --output_path ./benchmarks/max_norm/${task}/${model_name}/adaptive_none \
            --wandb_args project=lm-eval-angular-steering,entity= account \
            --log_samples <your >,id=${model_name}+${task}+adaptive_none+max_norm

        # steer
        for angle in $(seq 0 10 350); do
            echo "Evaluating model: $model_id with angle: $angle on port $port"
            lm_eval \
                --model local-completions \
                --tasks ${task} \
                --batch_size 1 \
                --model_args model=${model_id},base_url=http://0.0.0.0:${port}/angular_steering/${angle},num_concurrent=1,max_retries=3,tokenized_requests=False,max_length=4096 \
                --output_path ./benchmarks/max_norm/${task}/${model_name}/adaptive_${angle} \
                --wandb_args project=lm-eval-angular-steering,entity= account \
                --log_samples <your >,id=${model_name}+${task}+adaptive_${angle}+max_norm
        done
    done
done
