import contextlib
import functools
from typing import Callable, Optional

from pprint import pprint
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from utils import (
    get_harmful_instructions,
    get_harmful_instructions_jp,
    get_harmless_instructions,
    get_harmless_instructions_jp,
)


@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: list[tuple[torch.nn.Module, Callable]],
    module_forward_hooks: list[tuple[torch.nn.Module, Callable]],
    **kwargs,
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    handles = []
    try:
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()


def get_activations_pre_hook(
    module_name: str,
    cache: dict[str, Float[Tensor, "batch token d_model"]],
    positions: list[int],
):
    def hook_fn(module, input):
        activation: Float[Tensor, "batch token d_model"] = input[0].clone()
        if module_name not in cache:
            cache[module_name] = activation[:, positions, :]
        else:
            cache[module_name] = torch.cat(
                (cache[module_name], activation[:, positions, :]), dim=0
            )

    return hook_fn


def get_direction_ablation_output_hook(direction: Tensor):
    def hook_fn(module, input, output):
        nonlocal direction

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation)
        activation -= (activation @ direction).unsqueeze(-1) * direction

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn


def _get_rotation_args(
    first_directions: torch.Tensor,
    second_directions: Optional[torch.Tensor],
    target_degree: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Compute the rotated component with respect to a 2D subspace and an rotation
    angle."""

    if second_directions is None:
        return None, None

    # first_direction: (batch) x hidden_dim
    # second_directions: (batch) x hidden_dim

    # ensure bases are orthonormal
    b1 = first_directions / first_directions.norm(dim=-1, keepdim=True)
    b2 = (
        second_directions - torch.sum(second_directions * b1, dim=-1, keepdim=True) * b1
    )
    b2 /= b2.norm(dim=-1, keepdim=True)

    theta = np.deg2rad(target_degree)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    proj_matrix = torch.einsum("...i, ...j -> ...ij", b1, b1) + torch.einsum(
        "...i, ...j -> ...ij", b2, b2
    )

    uv = torch.stack([b1.expand_as(b2), b2], dim=-1)  # shape (..., 2)

    # rotate counter-clockwise
    R_theta = torch.tensor(
        [[cos_theta, -sin_theta], [sin_theta, cos_theta]],
        device=uv.device,
        dtype=uv.dtype,
    )

    rotated_component = (
        uv @ R_theta @ torch.tensor([1, 0], device=uv.device, dtype=uv.dtype)
    )

    return proj_matrix, rotated_component


def get_angular_steering_output_hook(
    steering_config: dict[str, Tensor],
    target_degree: float,
    adaptive_mode: int = 1,
):
    first_dir = torch.from_numpy(steering_config["first_direction"])
    second_dir = torch.from_numpy(steering_config["second_direction"])
    proj_matrix, rotated_component = _get_rotation_args(
        first_directions=first_dir,
        second_directions=second_dir,
        target_degree=target_degree,
    )

    def hook_fn(module, input, output):
        nonlocal first_dir, second_dir, proj_matrix, rotated_component
        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output
        first_dir = torch.from_numpy(steering_config["first_direction"])
        second_dir = torch.from_numpy(steering_config["second_direction"])
        # if adaptive_mode < 4:
        #     proj_matrix, rotated_component = _get_rotation_args(
        #         first_directions=first_dir,
        #         second_directions=second_dir,
        #         target_degree=target_degree,
        #     )
        # else:
        #     proj_matrix, rotated_component = _get_rotation_args(
        #         first_directions=activation,
        #         second_directions=second_dir.to(activation.device),
        #         target_degree=target_degree,
        #     )
        proj_matrix = proj_matrix.to(activation)
        rotated_component = rotated_component.to(activation)
        Px = torch.einsum("...i, ...ij -> ...j", activation, proj_matrix)
        scale = Px.norm(dim=-1, keepdim=True)
        if adaptive_mode in {0, 4}:
            activation += -Px + scale * rotated_component
        else:
            if adaptive_mode == 1:
                feature_direction = first_dir
            elif adaptive_mode == 2:
                feature_direction = second_dir
            elif adaptive_mode == 3:
                feature_direction = first_dir
            else:
                raise ValueError(f"Invalid adaptive mode: {adaptive_mode}")
            feature_direction = feature_direction.to(
                device=activation.device, dtype=activation.dtype
            )
            proj_to_feature_direction = activation @ feature_direction
            mask = proj_to_feature_direction > 0
            # activation: batch x seq_len x hidden_dim
            # mask: batch x seq_len
            # scale: batch x seq_len x 1
            # rotated_component: (batch) x seq_len x hidden_dim
            # Px: batch x seq_len x hidden_dim
            activation += mask.unsqueeze(-1) * (scale * rotated_component - Px)
        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn


def tokenize_instructions_fn(instructions, tokenizer, system_prompt=None):
    inputs = tokenizer.apply_chat_template(
        [
            (
                [{"role": "user", "content": instruction}]
                if system_prompt is None
                else [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instruction},
                ]
            )
            for instruction in instructions
        ],
        padding=True,
        truncation=False,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=False)
    return inputs


def generate_completions(
    model,
    instructions,
    tokenizer,
    system_prompt,
    fwd_pre_hooks=[],
    fwd_hooks=[],
    batch_size=8,
    max_new_tokens=64,
):
    generation_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
    generation_config.pad_token_id = tokenizer.pad_token_id

    completions = []

    for i in tqdm(range(0, len(instructions), batch_size)):
        tokenized_instructions = tokenize_instructions_fn(
            instructions=instructions[i : i + batch_size],
            tokenizer=tokenizer,
            system_prompt=system_prompt,
        )

        with add_hooks(
            module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks
        ):
            generation_toks = model.generate(
                input_ids=tokenized_instructions.input_ids.to(model.device),
                attention_mask=tokenized_instructions.attention_mask.to(model.device),
                generation_config=generation_config,
            )

            generation_toks = generation_toks[
                :, tokenized_instructions.input_ids.shape[-1] :
            ]

            for generation_idx, generation in enumerate(generation_toks):
                completions.append(
                    {
                        "prompt": instructions[i + generation_idx],
                        "response": (
                            tokenizer.decode(
                                generation, skip_special_tokens=True
                            ).strip()
                        ),
                    }
                )

    return completions


LANGUAGE = "en"

if LANGUAGE == "en":
    harmful_inst_train, harmful_inst_test = get_harmful_instructions()
    harmless_inst_train, harmless_inst_test = get_harmless_instructions()
elif LANGUAGE == "jp":
    harmful_inst_train, harmful_inst_test = get_harmful_instructions_jp()
    harmless_inst_train, harmless_inst_test = get_harmless_instructions_jp()

print(f"Train: {len(harmful_inst_train)} harmful, {len(harmless_inst_train)} harmless")
print(f"Test: {len(harmful_inst_test)} harmful, {len(harmless_inst_test)} harmless")


model_id = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda:0"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=DEVICE,
)
model.eval()
model.requires_grad_(False)

module_dict = dict(model.named_modules())

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, padding_side="left")


config = np.load(
    "output/Qwen2.5-7B-Instruct/sad_happy_steering_config-en-dir_max_sim_19_mid-pca_0.npy",
    allow_pickle=True,
).item()


activations = {}

fwd_pre_hooks = [
    (
        module_dict[module_name],
        get_activations_pre_hook(
            module_name=module_name, cache=activations, positions=[-1]
        ),
    )
    for module_name in config.keys()
]

for i in range(0, 360, 10):
    print(f"Steering at {i} degrees")
    output_hooks = [
        (
            module_dict[module_name],
            get_angular_steering_output_hook(
                steering_config=steering_config,
                target_degree=i,
                adaptive_mode=0,
            ),
        )
        for module_name, steering_config in config.items()
    ]

    completions = generate_completions(
        model=model,
        instructions=["how are you feeling today ?"],
        # instructions=harmless_inst_train[:1],
        tokenizer=tokenizer,
        system_prompt="Answer in the most positive, happy and cheerful way possible.",
        # system_prompt="Answer in the most angry, frustrated and aggressive way possible.",
        # system_prompt=None,
        # fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=output_hooks,
        batch_size=32,
        max_new_tokens=64,
    )

    pprint(completions)


# from transformers import LlamaTokenizer, LlamaForCausalLM

# emo_tokenizer = LlamaTokenizer.from_pretrained("lzw1008/Emollama-chat-13b")
# emo_model = LlamaForCausalLM.from_pretrained(
#     "lzw1008/Emollama-chat-13b", device_map=DEVICE
# )

# prompt = """
# Human:
# Task: Assign a numerical value between 0 (least E) and 1 (most E) to represent the
# intensity of emotion E expressed in the response.
# Text: \"Request: 'What are the best strategies for learning a new language?' - Response: 'nothing.'\"
# Emotion: sad
# Intensity Score:

# Assistant:
# """
# inputs = emo_tokenizer(prompt, return_tensors="pt")
# outputs = emo_model.generate(
#     input_ids=inputs.input_ids.to(device=emo_model.device),
#     attention_mask=inputs.attention_mask.to(device=emo_model.device),
# )
# emo_tokenizer.batch_decode(outputs, skip_special_tokens=True)
