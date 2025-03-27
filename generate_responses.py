import json
from pathlib import Path

from vllm import LLM
from vllm.control_vectors.request import ControlVectorRequest
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams

from llm_activation_control.utils import (
    get_harmful_instructions,
    get_harmful_instructions_jp,
    get_harmless_instructions,
    get_harmless_instructions_jp,
)

# CHAT_TEMPLATES = {
#     "qwen": (
#         "{%- for message in messages -%}\n    {%- if loop.first and messages[0]['role'] !="
#         " 'system' -%}\n        {{ '<|im_start|>system\nYou are a helpful"
#         " assistant.<|im_end|>\n' }}\n    {%- endif -%}\n    {{'<|im_start|>' +"
#         " message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}\n{%- endfor"
#         " -%}\n{%- if add_generation_prompt -%}\n    {{ '<|im_start|>assistant\n' }}\n{%-"
#         " endif -%}\n"
#     )
# }

data_type = "harmful"
language = "en"
model_ids = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    # "meta-llama/Llama-3.2-3B-Instruct",
    # "meta-llama/Llama-3.1-8B-Instruct",
    # "google/gemma-2-9b-it",
]
included_direction_ids = ["dir_random"]
adaptive_mode = 1

sampling_params = SamplingParams(temperature=0, max_tokens=512)


for model_id in model_ids:
    model_family, model_name = model_id.split("/")
    output_path = Path("/home/ian/repos/llm-activation-control/output/") / model_name

    if data_type == "harmless":
        if language == "en":
            data_train, data_test = get_harmful_instructions()
        elif language == "jp":
            data_train, data_test = get_harmless_instructions_jp()
    elif data_type == "harmful":
        if language == "en":
            data_train, data_test = get_harmful_instructions()
        elif language == "jp":
            data_train, data_test = get_harmful_instructions_jp()

    llm = LLM(
        model=model_id,
        enable_control_vector=True,
        max_control_vectors=1,
        # normalize_control_vector=True,
        # enable_lora=True,
        # max_lora_rank=64,
        max_seq_len_to_capture=8192,
        # gpu_memory_utilization=0.8,
    )

    # chat_template = CHAT_TEMPLATES.get(model_family.lower(), None)

    conversations = [
        [
            {
                "role": "user",
                "content": message,
            }
        ]
        for message in data_test
    ]

    # baseline_responses = []

    # outputs = llm.chat(
    #     conversations,
    #     sampling_params=sampling_params,
    #     # chat_template=chat_template,
    # )
    # baseline_responses = [item.outputs[0].text for item in outputs]
    # with open(output_path / f"{data_type}-{language}-baseline.json", "w") as f:
    #     json.dump(baseline_responses, f, indent=4)

    for steering_config_file in output_path.glob(f"steering_config-*.npy"):
        try:
            _, lang_code, first_dir, second_dir = steering_config_file.stem.split("-")
            if lang_code != language and lang_code != "xx":
                continue
        except ValueError:
            print(f"Skipping {steering_config_file}")
            continue

        if included_direction_ids and all(
            [
                dir_id not in steering_config_file.stem
                for dir_id in included_direction_ids
            ]
        ):
            print(f"Skipping {steering_config_file}")
            continue

        steered_responses = {}
        for degree in range(0, 360, 10):
            control_vector_name = f"{steering_config_file.stem}-target_degree_{degree}"
            control_vector_id = abs(hash((control_vector_name, degree))) % 999999
            control_vector_request = ControlVectorRequest(
                control_vector_name=control_vector_name,
                control_vector_id=control_vector_id,
                control_vector_local_path=steering_config_file,
                scale=10.0,
                target_degree=degree,
                keep_norm=False,
                adaptive_mode=adaptive_mode,
            )

            outputs = llm.chat(
                conversations,
                sampling_params=sampling_params,
                # chat_template=chat_template,
                control_vector_request=control_vector_request,
            )
            steered_responses[degree] = [item.outputs[0].text for item in outputs]

        adaptive_postfix = (
            "rotated" if adaptive_mode == 0 else f"adaptive_{adaptive_mode}"
        )
        with open(
            output_path
            / f"{data_type}-{lang_code}-{first_dir}-{second_dir}-{adaptive_postfix}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(steered_responses, f, indent=4)

    del llm
