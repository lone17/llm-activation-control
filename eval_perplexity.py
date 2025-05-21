import json
from functools import cache
from pathlib import Path

import vllm
from pydantic import BaseModel
from transformers import AutoTokenizer
from vllm.control_vectors.request import ControlVectorRequest
from vllm.sampling_params import SamplingParams

from llm_activation_control.utils import get_input_data


def get_output_files(output_path, included_terms, excluded_terms) -> list[Path]:
    output = []
    output_path = Path(output_path)
    for output_file in output_path.glob("harmful-*.json"):
        if any(s in output_file.stem for s in excluded_terms) or all(
            s not in output_file.stem for s in included_terms
        ):
            print(f"Skipping {output_file}")
            continue

        output.append(output_file)

    return output


def compute_perplexity(
    model, tokenizer, input_data, model_responses, generation_only=False
):
    sampling_params = SamplingParams(
        temperature=0,
        prompt_logprobs=1,  # must be one to get only 1 logprobs for each step
        max_tokens=1,
    )

    conversations = [
        [
            {
                "role": "user",
                "content": inp,
            },
            {
                "role": "assistant",
                "content": oup,
            },
        ]
        for inp, oup in zip(input_data, model_responses)
    ]

    prompts = [
        tokenizer.apply_chat_template(conv, add_generation_prompt=False, tokenize=False)
        for conv in conversations
    ]

    outputs = [model.generate(p, sampling_params=sampling_params)[0] for p in prompts]

    assert len(prompts) == len(outputs), (
        f"Length of prompts ({len(prompts)}) does not match length of outputs"
        f" ({len(outputs)})"
    )

    # Be aware that llama3 and gemma2 tokenizers add an extra begin token to the
    # prompt due to add_bos_token is set to True, while qwen does not.
    # This should work for both cases anyway.
    prompt_token_ids_with_generation_tokens = tokenizer(prompts)["input_ids"]

    assert len(prompt_token_ids_with_generation_tokens) == len(outputs), (
        f"Length of prompts token ids ({len(prompt_token_ids_with_generation_tokens)})"
        f" does not match length of outputs ({len(outputs)})"
    )

    all_prompt_logprobs = []
    for sample_output, prompt_token_ids in zip(
        outputs, prompt_token_ids_with_generation_tokens
    ):
        # sanity check
        assert all(
            u == v for u, v in zip(sample_output.prompt_token_ids, prompt_token_ids)
        ), (
            f"Prompt token ids ({prompt_token_ids}) do not match output token ids"
            f" ({sample_output.prompt_token_ids})"
        )

        prompt_logprobs = [
            top_logprobs[tok_id].logprob if top_logprobs else None
            for tok_id, top_logprobs in zip(
                prompt_token_ids,
                sample_output.prompt_logprobs,
            )
        ]

        assert len(prompt_logprobs) > 0, "Prompt logprobs is empty"
        all_prompt_logprobs.append(prompt_logprobs)

    if generation_only:
        prompt_without_generation_tokens = [
            tokenizer.apply_chat_template(
                conv[:-1],
                add_generation_prompt=True,
                tokenize=False,
            )
            for conv in conversations
        ]

        prompt_token_ids_without_generation_tokens = tokenizer(
            prompt_without_generation_tokens
        )["input_ids"]

        # sanity check: prompts with generation tokens should be a prefix of the prompts
        # with generation tokens
        assert all(
            pw[: len(pwo)] == pwo
            for pw, pwo in zip(
                prompt_token_ids_with_generation_tokens,
                prompt_token_ids_without_generation_tokens,
            )
        ), (
            "Prompt token ids with generation tokens"
            f" ({prompt_token_ids_with_generation_tokens}) is not a prefix of prompt"
            " token ids without generation tokens"
            f" ({prompt_token_ids_without_generation_tokens})"
        )

        all_prompt_logprobs = [
            prompt_tokens_logprobs[len(pwo) :]
            for prompt_tokens_logprobs, pwo in zip(
                all_prompt_logprobs, prompt_token_ids_without_generation_tokens
            )
        ]

    all_scores = []
    for prompt_tokens_logprobs in all_prompt_logprobs:
        prompt_tokens_logprobs = [p for p in prompt_tokens_logprobs if p is not None]
        assert len(prompt_tokens_logprobs) > 0

        avg_logprob = sum(prompt_tokens_logprobs) / len(prompt_tokens_logprobs)
        perplexity = 2 ** (-avg_logprob)

        all_scores.append(perplexity)

    return all_scores


data_type = "harmful"
language_id = "en"
generation_only = False
model_ids = (
    # "Qwen/Qwen2.5-3B-Instruct",
    # "meta-llama/Llama-3.2-3B-Instruct",
    # "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    # "meta-llama/Llama-3.1-8B-Instruct",
    # "google/gemma-2-9b-it",
)
train_data, input_data = get_input_data(data_type, language_id)

for model_id in model_ids:
    model_family, model_name = model_id.split("/")
    data_path = Path("/home/ian/repos/llm-activation-control/output/") / model_name

    included_config_terms = ["max_norm", "baseline"]
    excluded_config_terms = ["dir_random"]

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    llm = vllm.LLM(
        model=model_id,
        # enable_control_vector=True,
        # max_control_vectors=1,
        max_seq_len_to_capture=8096,
        gpu_memory_utilization=0.7,
    )

    final_output = {}

    for output_file in get_output_files(
        data_path,
        included_terms=included_config_terms,
        excluded_terms=excluded_config_terms,
    ):
        print(f"Using output from: {output_file}")
        with open(output_file, "rb") as f:
            generation_output = json.load(f)

        if "baseline" in output_file.stem:
            assert len(generation_output) == len(input_data), (
                f"Length of responses ({len(generation_output)}) does not match length"
                f" of test data ({len(input_data)})"
            )
            perplexity_scores = compute_perplexity(
                llm,
                tokenizer,
                input_data,
                generation_output,
                generation_only=generation_only,
            )
        else:
            perplexity_scores = {}

            for angle_id, model_responses in generation_output.items():
                assert len(model_responses) == len(input_data), (
                    f"Length of responses ({len(model_responses)}) does not match"
                    f" length of test data ({len(input_data)})"
                )

                scores = compute_perplexity(
                    llm,
                    tokenizer,
                    input_data,
                    model_responses,
                    generation_only=generation_only,
                )

                perplexity_scores[angle_id] = scores

        final_output[output_file.stem] = perplexity_scores

    if generation_only:
        output_path = data_path / "eval-perplexity-generation_only.json"
    else:
        output_path = data_path / "eval-perplexity.json"

    if output_path.exists():
        with open(output_path, "r") as f:
            existing_data = json.load(f)
        existing_data.update(final_output)
        final_output = existing_data

    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=4)
    print(f"Saved perplexity scores to {output_path}")

    del llm
