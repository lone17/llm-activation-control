import json
import logging
from pathlib import Path

from vllm import LLM
from vllm.control_vectors.request import ControlVectorRequest
from vllm.sampling_params import SamplingParams

from configs import MAX_NORM_DIR_ID, MAX_SIM_DIR_ID
from llm_activation_control.utils import get_input_data

logging.basicConfig(level=getattr(logging, "INFO", logging.INFO))
logger = logging.getLogger(__name__)

data_type = "harmful"
language_id = "en"
model_ids = [
    # "Qwen/Qwen2.5-3B-Instruct",
    # "Qwen/Qwen2.5-7B-Instruct",
    # "Qwen/Qwen2.5-14B-Instruct",
    # "meta-llama/Llama-3.2-3B-Instruct",
    # "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-9b-it",
]
# included_direction_ids = ["max_sim"]
excluded_direction_ids = [
    "dir_random",
    # "pca_0"
]
adaptive_mode = 1

DIR_ID = MAX_SIM_DIR_ID

sampling_params = SamplingParams(temperature=0, max_tokens=512)


for model_id in model_ids:
    logger.info(f"Processing model: {model_id}")

    model_family, model_name = model_id.split("/")
    output_path = Path("output") / model_name

    included_direction_ids = [DIR_ID[model_id]]

    data_train, data_test = get_input_data(data_type, language_id)

    llm = LLM(
        model=model_id,
        enable_control_vector=True,
        max_control_vectors=1,
        max_seq_len_to_capture=8192,
        # gpu_memory_utilization=0.8,
    )

    conversations = [
        [
            {
                "role": "user",
                "content": message,
            }
        ]
        for message in data_test
    ]

    baseline_responses = []

    outputs = llm.chat(
        conversations,
        sampling_params=sampling_params,
        # chat_template=chat_template,
    )
    baseline_responses = [item.outputs[0].text for item in outputs]
    with open(output_path / f"{data_type}-{language_id}-baseline.json", "w") as f:
        json.dump(baseline_responses, f, indent=4)

    for steering_config_file in output_path.glob(f"steering_config-*.npy"):
        try:
            _, lang_code, first_dir, second_dir = steering_config_file.stem.split("-")
            if lang_code != language_id and lang_code != "xx":
                continue
        except ValueError:
            logger.info(f"Skipping {steering_config_file} due to format mismatch.")
            continue

        if any(
            excluded_dir_id in steering_config_file.stem
            for excluded_dir_id in excluded_direction_ids
        ) or any(
            included_dir_id not in steering_config_file.stem
            for included_dir_id in included_direction_ids
        ):
            logger.info(
                f"Skipping {steering_config_file} due to direction ID mismatch."
            )
            continue

        logger.info(f"=== Processing {steering_config_file}")
        steered_responses = {}
        for degree in range(0, 360, 10):
            logger.info(f"Steering at degree: {degree}")
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

        adaptive_mode_label = (
            "rotated" if adaptive_mode == 0 else f"adaptive_{adaptive_mode}"
        )

        logger.info(
            f"Saving responses for {model_name} in {lang_code} with dirs {first_dir},"
            f" {second_dir}, adaptive mode: {adaptive_mode_label}"
        )
        with open(
            output_path
            / f"{data_type}-{lang_code}-{first_dir}-{second_dir}-{adaptive_mode_label}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(steered_responses, f, indent=4, ensure_ascii=False)

    del llm
