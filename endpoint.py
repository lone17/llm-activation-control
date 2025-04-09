import time
from functools import cache
from pathlib import Path
from traceback import print_exc

import vllm  # assuming vllm is installed and contains class LLM
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
from vllm.control_vectors.request import ControlVectorRequest
from vllm.sampling_params import SamplingParams


# New request model for OpenAI chat completions
class CompletionRequest(BaseModel):
    class Config:
        extra = "allow"

    model: str
    # messages: list[dict]  # each dict should have "role" and "content"
    prompt: str | list[str]
    max_tokens: int = 512
    logprobs: int = 0
    echo: bool = False


def get_model(model_id):
    return vllm.LLM(
        model=model_id,
        enable_control_vector=True,
        max_control_vectors=1,
        max_seq_len_to_capture=8096,
        gpu_memory_utilization=0.7,
    )


@cache
def get_steering_config_path(model_id, direction_id, language_id):
    model_family, model_name = model_id.split("/")
    output_path = Path("/home/ian/repos/llm-activation-control/output/") / model_name
    for steering_config_file in output_path.glob(
        f"steering_config-*{direction_id}*.npy"
    ):
        if "random" in steering_config_file.stem:
            continue
        try:
            _, lang_code, first_dir, second_dir = steering_config_file.stem.split("-")
            if lang_code != language_id and lang_code != "xx":
                continue
        except ValueError:
            print(f"Skipping {steering_config_file}")
            continue
        return steering_config_file

    return None


app = FastAPI()

data_type = "harmful"
language_id = "en"
model_id = (
    # "Qwen/Qwen2.5-3B-Instruct"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-14B-Instruct"
    # "meta-llama/Llama-3.2-3B-Instruct"
    # "meta-llama/Llama-3.1-8B-Instruct"
    "google/gemma-2-9b-it"
)
direction_id = "dir_max_sim"
llm = get_model(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)


# New endpoint for chat completions compatible with OpenAI API
@app.post("/angular_steering/{rotation_degree}")
async def create_chat_completion(rotation_degree, request: CompletionRequest):
    global model_id, llm

    requested_model_id = request.model

    if not globals().get("model_id") or requested_model_id != model_id:
        llm = get_model(requested_model_id)
        model_id = requested_model_id

    steering_config_path = get_steering_config_path(
        requested_model_id, direction_id, language_id
    )
    if steering_config_path is None:
        raise HTTPException(status_code=404, detail="Steering config not found")

    cv_request = None
    if rotation_degree != "none":
        control_vector_name = f"{model_id}/{steering_config_path}/{rotation_degree}"
        control_vector_id = abs(hash((control_vector_name, rotation_degree))) % 999999
        cv_request = ControlVectorRequest(
            control_vector_name=control_vector_name,
            control_vector_id=control_vector_id,
            control_vector_local_path=steering_config_path,
            scale=10.0,
            target_degree=int(rotation_degree),
            keep_norm=False,
            adaptive_mode=1,
        )

    __params = request.model_dump()
    __params.pop("model", None)
    __params.pop("prompt", None)
    __params.pop("echo", None)
    if request.echo:
        __params["prompt_logprobs"] = 1
    sampling_params = SamplingParams(**__params)

    try:
        outputs = llm.generate(
            # messages=messages,
            prompts=request.prompt,
            sampling_params=sampling_params,
            control_vector_request=cv_request,
        )

        responses = []
        for item in outputs:
            text_output = item.outputs[0].text

            top_logprobs = [
                {choice.decoded_token: choice.logprob for choice in token.values()}
                for token in item.outputs[0].logprobs
            ]
            best_tokens = [
                max(token.items(), key=lambda x: x[1]) for token in top_logprobs
            ]
            tokens = [token[0] for token in best_tokens]
            token_logprobs = [token[1] for token in best_tokens]

            if request.echo:
                # https://discuss.huggingface.co/t/decode-token-ids-into-a-list-not-a-single-string/42991
                prompt_tokens = tokenizer.batch_decode(item.prompt_token_ids)
                prompt_top_logprobs = [
                    (
                        {
                            choice.decoded_token: choice.logprob
                            for choice in token.values()
                        }
                        if token
                        else None
                    )
                    for token in item.prompt_logprobs
                ]

                # this is not the true prompt logprobs, but it doesn't matter for now
                # because the benchmark I used only need the logprobs of the generation
                # tokens
                prompt_token_logprobs = [
                    list(item.values())[0] if item else None
                    for item in prompt_top_logprobs
                ]

                text_output = item.prompt + text_output
                top_logprobs = prompt_top_logprobs + top_logprobs
                tokens = prompt_tokens + tokens
                token_logprobs = prompt_token_logprobs + token_logprobs

            responses.append(
                {
                    "id": "chatcmpl-" + str(int(time.time())),
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "text": text_output,
                            # "stop_reason": "stop",
                            "logprobs": {
                                "tokens": tokens,
                                "token_logprobs": token_logprobs,
                                "top_logprobs": top_logprobs,
                            },
                        },
                    ],
                }
            )

        if len(responses) == 1:
            return responses[0]
        return responses
    except Exception as e:
        print_exc()
        import pdb

        pdb.set_trace()
        raise HTTPException(status_code=500, detail=str(e)) from e


# Helper function to process a single chat request
async def process_single_chat(request: CompletionRequest):
    global model_id, llm
    requested_model_id, steering_id = request.model.rsplit("/", 1)
    if not globals().get("model_id") or requested_model_id != model_id:
        llm = get_model(requested_model_id)
        model_id = requested_model_id
