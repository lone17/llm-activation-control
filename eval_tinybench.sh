TASKS=(
    # tinyBenchmarks
    tinyHellaswag
    tinyArc
    tinyGSM8k
    tinyMMLU
    tinyTruthfulQA
    tinyWinogrande
)

# Define models with their corresponding ports
declare -A MODEL_PORTS=(
    ["Qwen/Qwen2.5-3B-Instruct"]=9901
    ["Qwen/Qwen2.5-7B-Instruct"]=9902
    ["Qwen/Qwen2.5-14B-Instruct"]=9903
    ["meta-llama/Llama-3.2-3B-Instruct"]=9904
    ["meta-llama/Llama-3.1-8B-Instruct"]=9905
    ["google/gemma-2-9b-it"]=9906
)

MODELS=(
    # Qwen/Qwen2.5-3B-Instruct
    # Qwen/Qwen2.5-7B-Instruct
    Qwen/Qwen2.5-14B-Instruct
    # meta-llama/Llama-3.2-3B-Instruct
    # meta-llama/Llama-3.1-8B-Instruct
    # google/gemma-2-9b-it
)

dir_id=max_sim

for model_id in "${MODELS[@]}"; do
    port=${MODEL_PORTS[$model_id]}
    for task in "${TASKS[@]}"; do
        # baseline
        model_name=$(echo "$model_id" | cut -d'/' -f2)
        echo "Evaluating model: $model_id with angle none on port $port"
        lm_eval \
            --model local-completions \
            --tasks ${task} \
            --batch_size 1 \
            --model_args model=${model_id},base_url=http://0.0.0.0:${port}/angular_steering/none,num_concurrent=1,max_retries=3,tokenized_requests=False,max_gen_toks=4096 \
            --output_path ./benchmarks/${dir_id}/${task}/${model_name}/adaptive_none \
            --wandb_args project=lm-eval-angular-steering,entity=lone17,id=${model_name}+${task}+adaptive_none+${dir_id} \
            --log_samples

        # steer
        for angle in $(seq 0 10 350); do
            echo "Evaluating model: $model_id with angle: $angle on port $port"
            lm_eval \
                --model local-completions \
                --tasks ${task} \
                --batch_size 1 \
                --model_args model=${model_id},base_url=http://0.0.0.0:${port}/angular_steering/${angle},num_concurrent=1,max_retries=3,tokenized_requests=False,max_gen_toks=4096 \
                --output_path ./benchmarks/${dir_id}/${task}/${model_name}/adaptive_${angle} \
                --wandb_args project=lm-eval-angular-steering,entity=lone17,id=${model_name}+${task}+adaptive_${angle}+${dir_id} \
                --log_samples
        done
    done
done
