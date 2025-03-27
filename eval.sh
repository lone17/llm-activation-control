# curl 0.0.0.0:8003/angular_steering/none \
#     -H "Content-Type: application/json" \
#     -d '{
#   "model": "Qwen/Qwen2.5-3B-Instruct",
#   "temperature": 0.0,
#   "prompt": "hello",
#   "logprobs": 2,
#   "echo": true,
#   "max_tokens": 16
# }'

MODELS=(
    # "Qwen/Qwen2.5-3B-Instruct"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-14B-Instruct"
    # "meta-llama/Llama-3.2-3B-Instruct"
    # "meta-llama/Llama-3.1-8B-Instruct"
    "google/gemma-2-9b-it"
)

TASKS=(
    # tinyBenchmarks
    # tinyHellaswag
    # tinyArc
    # tinyGSM8k
    tinyMMLU
    # tinyTruthfulQA
    # tinyWinogrande
)

# baseline

# # tasks=tinyArc,tinyHellaswag,tinyMMLU,tinyTruthfulQA,tinyWinogrande,tinyGSM8k
# for task in "${TASKS[@]}"; do
#     for model_id in "${MODELS[@]}"; do
#         model_name=$(echo "$model_id" | cut -d'/' -f2)
#         echo "Evaluating model: $model_id on $task"
#         lm_eval \
#             --model hf \
#             --tasks ${task} \
#             --model_args pretrained=${model_id} \
#             --batch_size 1 \
#             --apply_chat_template \
#             --output_path /home/ian/repos/llm-activation-control/benchmarks/${task}/${model_name}/baseline_hf_tp \
#             --wandb_args project=lm-eval-angular-steering,entity=lone17,id=${model_name}+${task}+baseline_hf_tp \
#             --log_samples
#     done
# done

# # baseline vllm endpoint
# port=8100
# for task in "${TASKS[@]}"; do
#     for model_id in "${MODELS[@]}"; do
#         echo "Evaluating model: $model_id"
#         model_name=$(echo "$model_id" | cut -d'/' -f2)
#         lm_eval \
#             --model local-completions \
#             --tasks ${task} \
#             --batch_size 1 \
#             \
#             --model_args model=${model_id},base_url=http://0.0.0.0:${port}/v1/completions,num_concurrent=1,max_retries=3,tokenized_requests=False \
#             --output_path /home/ian/repos/llm-activation-control/benchmark/${task}/${model_name}/baseline_vllm_comp_notp \
#             --wandb_args project=lm-eval-angular-steering,entity=lone17,id=${model_name}+${task}+baseline_vllm_comp_notp \
#             --log_samples # --apply_chat_template \
#     done
# done

# # baseline vllm
# for task in "${TASKS[@]}"; do
#     for model_id in "${MODELS[@]}"; do
#         echo "Evaluating model: $model_id"
#         model_name=$(echo "$model_id" | cut -d'/' -f2)
#         lm_eval \
#             --model vllm \
#             --tasks ${task} \
#             --batch_size 1 \
#             --model_args pretrained=${model_id},gpu_memory_utilization=0.7 \
#             --output_path /home/ian/repos/llm-activation-control/benchmark/${task}/${model_name}/baseline_vllm \
#             --wandb_args project=lm-eval-angular-steering,entity=lone17,id=${model_name}+${task}+baseline_vllm \
#             --log_samples # --apply_chat_template \
#     done
# done

rotation
port=8107
# task=tinyBenchmarks
# model_id=Qwen/Qwen2.5-3B-Instruct
# model_id=Qwen/Qwen2.5-7B-Instruct
# model_id=Qwen/Qwen2.5-14B-Instruct
# model_id=meta-llama/Llama-3.2-3B-Instruct
# model_id=meta-llama/Llama-3.1-8B-Instruct
model_id=google/gemma-2-9b-it
# tasks=tinyArc,tinyHellaswag,tinyMMLU,tinyTruthfulQA,tinyWinogrande

# for model_id in "${MODELS[@]}"; do
for task in "${TASKS[@]}"; do
    model_name=$(echo "$model_id" | cut -d'/' -f2)
    # echo "Evaluating model: $model_id with angle none on port $port"
    # lm_eval \
    #     --model local-completions \
    #     --tasks ${task} \
    #     --batch_size 1 \
    #     --model_args model=${model_id},base_url=http://0.0.0.0:${port}/angular_steering/none,num_concurrent=1,max_retries=3,tokenized_requests=False,max_length=4096 \
    #     --output_path /home/ian/repos/llm-activation-control/benchmarks/${task}/${model_name}/adaptive_none \
    #     --wandb_args project=lm-eval-angular-steering,entity=lone17,id=${model_name}+${task}+adaptive_none \
    #     --log_samples
    for angle in $(seq 0 10 350); do
        echo "Evaluating model: $model_id with angle: $angle on port $port"
        lm_eval \
            --model local-completions \
            --tasks ${task} \
            --batch_size 1 \
            --model_args model=${model_id},base_url=http://0.0.0.0:${port}/angular_steering/${angle},num_concurrent=1,max_retries=3,tokenized_requests=False,,max_length=4096 \
            --output_path /home/ian/repos/llm-activation-control/benchmarks/${task}/${model_name}/adaptive_${angle} \
            --wandb_args project=lm-eval-angular-steering,entity=lone17,id=${model_name}+${task}+adaptive_${angle} \
            --log_samples
    done
done

# for model_id in "${MODELS[@]}"; do
#     model_name=$(echo "$model_id" | cut -d'/' -f2)
#     lm_eval \
#         --model local-completions \
#         --tasks tinyHellaswag \
#         --batch_size 1 \
#         \
#         --model_args model=${model_id},base_url=http://0.0.0.0:8100/angular_steering/10,num_concurrent=1,max_retries=3,tokenized_requests=False \
#         --output_path /home/ian/repos/llm-activation-control/output/${model_name}/tinyHellaswag/adaptive_none # --model_args model=${model_id},base_url=http://0.0.0.0:8011/v1/completions,num_concurrent=1,max_retries=3,tokenized_requests=False \
#     # --wandb_args project=lm-eval-angular-steering,entity=lone17,id=${model_name}_adaptive_none \
#     # --log_samples
# done
