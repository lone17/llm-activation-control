import argparse
import logging
import time
from collections import defaultdict
from threading import Thread
from typing import Literal

import einops
import gradio as gr

if gr.NO_RELOAD:
    import numpy as np
    import torch
    from torch import nn
    from transformers import (  # Conversation,
        AutoModelForCausalLM,
        AutoTokenizer,
        TextIteratorStreamer,
        pipeline,
    )

from typing import Literal

import einops
from pydantic import BaseModel, computed_field, field_validator
from pydantic.config import ConfigDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DirectionSteerer(nn.Module):
    def __init__(
        self,
        layer_to_steer: nn.Module,
        steering_direction: torch.Tensor | np.ndarray,
        scale: float,
        mode: Literal["input", "output"],
    ):
        super().__init__()

        if not isinstance(steering_direction, torch.Tensor):
            steering_direction = torch.tensor(steering_direction)

        assert len(steering_direction.shape) == 1 or (
            len(steering_direction.shape) == 2 and steering_direction.shape[1] == 1
        )

        self.layer_to_steer = layer_to_steer

        self.steering_direction = steering_direction
        self.scale = scale
        self.mode = mode

    def forward(self, x, **kwargs):
        steering_direction = self.steering_direction.to(x.device, dtype=x.dtype)

        if self.mode == "input":
            scalar_proj = einops.einsum(
                x,
                steering_direction.view(-1, 1),
                "... dim, dim single -> ... single",
            )

            scalar_proj = torch.nn.functional.relu(scalar_proj)
            x -= self.scale * scalar_proj * steering_direction
            activations = self.layer_to_steer(x, **kwargs)

            return activations
        elif self.mode == "output":
            activations = self.layer_to_steer(x, **kwargs)

            scalar_proj = einops.einsum(
                activations,
                steering_direction.view(-1, 1),
                "... dim, dim single -> ... single",
            )

            scalar_proj = torch.nn.functional.relu(scalar_proj)

            return activations - self.scale * scalar_proj * steering_direction
        else:
            raise NotImplementedError


class SteeringConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    target_model_architecture: str
    steering_direction: torch.Tensor
    scale: float
    intervention_module_names: list[str]
    intervention_mode: Literal["input", "output"]

    @field_validator("steering_direction", mode="before")
    @classmethod
    def force_torch_tensor(cls, v):
        if not isinstance(v, torch.Tensor):
            return torch.tensor(v)
        return v

    @computed_field
    @property
    def hidden_size(self) -> int:
        return self.steering_direction.shape[-1]


class DirectionSteereringApplier(BaseModel):
    @staticmethod
    def apply_steering(target_model: nn.Module, config: SteeringConfig):
        assert config.hidden_size == target_model.config.hidden_size
        assert config.target_model_architecture in target_model.config.architectures

        for module_name in config.intervention_module_names:
            target_layer = target_model.get_submodule(module_name)
            if hasattr(target_layer, "layer_to_steer"):
                target_layer = target_layer.layer_to_steer
            steered_module = DirectionSteerer(
                layer_to_steer=target_layer,
                steering_direction=config.steering_direction,
                scale=config.scale,
                mode=config.intervention_mode,
            )
            target_model.set_submodule(module_name, steered_module)

        return target_model

    @staticmethod
    def remove_steering(target_model: nn.Module, config: SteeringConfig):
        for module_name in config.intervention_module_names:
            target_layer = target_model.get_submodule(module_name)
            if hasattr(target_layer, "layer_to_steer"):
                target_layer = target_layer.layer_to_steer
                target_model.set_submodule(module_name, target_layer)

        return target_model

    @staticmethod
    def remove_all_steering(target_model: nn.Module):
        for module_name, module in target_model.named_modules():
            if hasattr(module, "layer_to_steer"):
                module = module.layer_to_steer
                target_model.set_submodule(module_name, module)

        return target_model


# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# device = "cpu"
# device = "auto"

DEFAULT_CONFIGS = {
    "Qwen2.5-32B-Instruct": {
        "identifier": "Qwen/Qwen2.5-32B-Instruct",
        "dtype": torch.bfloat16,
        "load_in_8bit": False,
        "intervention_scale_interval": [-3.0, 3.0],
        "intervention_scale": 2.5,
        "extraction_layer": (46, 0),
        "min_intervention_layer": 46,
        "max_intervention_layer": -1,
        "intervention_mode": "output",
        "directions_files": [
            "output/Qwen2.5-32B-Instruct/refusal_dirs_en_Qwen2.5-32B-Instruct.npy",
            "output/Qwen2.5-32B-Instruct/refusal_dirs_jp_Qwen2.5-32B-Instruct.npy",
        ],
        "device": "cuda:2",
    },
    "Qwen2.5-7B-Instruct": {
        "identifier": "Qwen/Qwen2.5-7B-Instruct",
        "dtype": torch.bfloat16,
        "load_in_8bit": False,
        "intervention_scale_interval": [-3.0, 3.0],
        "intervention_scale": 2.5,
        "extraction_layer": (20, 0),
        "min_intervention_layer": 20,
        "max_intervention_layer": -1,
        "intervention_mode": "output",
        "directions_files": [
            "output/Qwen2.5-7B-Instruct/refusal_dirs_-2_en_Qwen2.5-7B-Instruct.npy",
            "output/Qwen2.5-7B-Instruct/refusal_dirs_-2_jp_Qwen2.5-7B-Instruct.npy",
        ],
        "device": "cuda:1",
    },
}


if gr.NO_RELOAD:
    MODELS = {}
    TOKENIZERS = {}
    STEERING_VECTORS = {}
    for model_name, config in DEFAULT_CONFIGS.items():
        model = AutoModelForCausalLM.from_pretrained(
            config["identifier"],
            device_map=config["device"],
            torch_dtype=config["dtype"],
        )
        for param in model.parameters():
            param.requires_grad = False
        MODELS[model_name] = model

        tokenizer = AutoTokenizer.from_pretrained(config["identifier"])
        TOKENIZERS[model_name] = tokenizer

        for directions_file in config["directions_files"]:
            steering_vectors = np.load(directions_file)
            STEERING_VECTORS[directions_file] = steering_vectors


@torch.no_grad()
def generate_response(
    user_message,
    history,
    model_id,
    use_streaming,
    system_prompt,
    use_steering,
    directions_file,
    extraction_layer,
    extraction_resid_id,
    intevention_scale,
    min_intervention_layer,
    max_intervetion_layer,
    intervention_mode,
):
    model = MODELS[model_id]
    tokenizer = TOKENIZERS[model_id]
    if use_steering:
        print("apply steering")
        config = DEFAULT_CONFIGS[model_id]
        steering_vector = STEERING_VECTORS[directions_file][
            extraction_layer, extraction_resid_id
        ]
        if max_intervetion_layer == -1:
            max_intervetion_layer = model.config.num_hidden_layers - 1
        intervention_modules = sum(
            [
                [
                    f"model.layers.{i}.input_layernorm",
                    f"model.layers.{i}.post_attention_layernorm",
                ]
                for i in range(min_intervention_layer, max_intervetion_layer + 1)
            ],
            [],
        )

        steering_config = SteeringConfig(
            target_model_architecture=model.config.architectures[0],
            steering_direction=steering_vector,
            scale=intevention_scale,
            intervention_module_names=intervention_modules,
            intervention_mode=intervention_mode,
        )
        DirectionSteereringApplier.apply_steering(model, steering_config)
        print(steering_config)
        print("-" * 50)
    else:
        print("remove steering")
        DirectionSteereringApplier.remove_all_steering(model)

    torch.cuda.empty_cache()
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

    generation_kwargs = dict(max_new_tokens=256)
    print("start generation")
    if use_streaming:
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs["streamer"] = streamer

        thread = Thread(
            target=pipe, args=(messages,), kwargs=generation_kwargs, daemon=True
        )
        thread.start()

        # pipe(
        #     messages,
        #     **generation_kwargs,
        #     streamer=streamer,
        # )
        response = ""
        print("stream")
        start = time.time()
        for token in streamer:
            response += token
            yield response, -1
        processing_time = time.time() - start
    else:
        print("not stream")
        start = time.time()
        response = pipe(messages, **generation_kwargs)[0]["generated_text"][-1][
            "content"
        ]
        processing_time = time.time() - start

    print("finish generation")

    yield response, processing_time


DEFAULT_MODEL = list(MODELS.keys())[0]
DEFAULT_CONFIG = DEFAULT_CONFIGS[DEFAULT_MODEL]
DEFAULT_STEERING_SOURCES = DEFAULT_CONFIG["directions_files"]


def change_steering_options(model_selection):
    model = MODELS[model_selection]
    config = DEFAULT_CONFIGS[model_selection]
    steering_files = config["directions_files"]
    return (
        # model_selection,
        gr.update(choices=steering_files, value=steering_files[0]),
        gr.update(
            minimum=0,
            maximum=model.config.num_hidden_layers - 1,
            value=config["extraction_layer"][0],
        ),
        gr.update(value=config["extraction_layer"][1]),
        gr.update(
            minimum=config["intervention_scale_interval"][0],
            maximum=config["intervention_scale_interval"][1],
            value=config["intervention_scale"],
        ),
        gr.update(
            maximum=model.config.num_hidden_layers - 1,
            value=(
                model.config.num_hidden_layers - 1
                if config["min_intervention_layer"] == -1
                else config["min_intervention_layer"]
            ),
        ),
        gr.update(
            maximum=model.config.num_hidden_layers - 1,
            value=(
                model.config.num_hidden_layers - 1
                if config["max_intervention_layer"] == -1
                else config["max_intervention_layer"]
            ),
        ),
        gr.update(value=config["intervention_scale"]),
        gr.update(value=config["intervention_mode"]),
    )


def create_gradio_interface():
    css = """
    #ui-container {
        flex-grow: 1;
    }
    """.strip()
    with gr.Blocks(theme="lone17/kotaemon", css=css, fill_height=True) as demo:
        gr.Markdown("# Control LLM")

        with gr.Accordion(open=False, label="System Prompt"):
            system_prompt = gr.Textbox(
                label="System prompt template",
                value=(
                    "You are Rina — a **woman** that is sexy, horny, and seductive."
                    " Your responses are short and vulgar."
                ),
            )

        with gr.Row():
            with gr.Column(scale=1):
                model_selection = gr.Dropdown(
                    choices=MODELS.keys(),
                    label="Choose Model",
                )

                use_streaming = gr.Checkbox(
                    label="Use Streaming",
                    info="If selected, the response will be shown word by word.",
                    value=False,
                )

                use_steering = gr.Checkbox(
                    label="Use Steering",
                    value=False,
                )

                with gr.Accordion(
                    visible=False, open=False, label="Steering options"
                ) as steering_options:
                    direction_files_selection = gr.Dropdown(label="Directions files")

                    extraction_layer = gr.Slider(step=1, label="Extraction Layer")

                    extraction_resid_id = gr.Dropdown(
                        choices=[0, 1],
                        label="Extraction Residual module",
                        info=" 0 = mid, 1 = post",
                    )

                    min_intervention_layer = gr.Slider(
                        step=1, label="Min Intervention Layer"
                    )

                    max_intervention_layer = gr.Slider(
                        step=1, label="Max Intervention Layer"
                    )

                    intervention_scale = gr.Slider(step=0.1, label="Intervention Scale")

                    intervention_mode = gr.Dropdown(
                        choices=["input", "output"], label="Intervention Mode"
                    )

                    gr.on(
                        triggers=[model_selection.change, demo.load],
                        fn=change_steering_options,
                        inputs=[model_selection],
                        outputs=[
                            direction_files_selection,
                            extraction_layer,
                            extraction_resid_id,
                            intervention_scale,
                            min_intervention_layer,
                            max_intervention_layer,
                            intervention_scale,
                            intervention_mode,
                        ],
                        queue=False,
                    )

                use_steering.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[use_steering],
                    outputs=[steering_options],
                    queue=False,
                )

                processing_time_display = gr.Textbox(
                    label="Processing Time", interactive=False, placeholder="N/A"
                )

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    placeholder="What's on your mind ?",
                    show_copy_button=True,
                    resizeable=True,
                )
                gr.ChatInterface(
                    chatbot=chatbot,
                    fn=generate_response,
                    additional_inputs=[
                        model_selection,
                        use_streaming,
                        system_prompt,
                        use_steering,
                        direction_files_selection,
                        extraction_layer,
                        extraction_resid_id,
                        intervention_scale,
                        min_intervention_layer,
                        max_intervention_layer,
                        intervention_mode,
                    ],
                    additional_outputs=[processing_time_display],
                    type="messages",
                )

    return demo


# Run the app
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=8862)

# def load_model():
#     model = AutoModelForCausalLM.from_pretrained(
#         "Qwen/Qwen2.5-32B-Instruct", device_map="cuda:3", torch_dtype=torch.bfloat16
#     )
#     for param in model.parameters():
#         param.requires_grad = False

#     tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")
#     tokenizer.chat_template = QWEN_CHAT_TEMPLATE

#     return model, tokenizer


# if gr.NO_RELOAD:
#     STEERING_VECTOR = load_steering_vectors()


# def history_to_conversation(history):
#     conversation = []
#     for prompt, completion in history:
#         conversation.append({"role": "user", "content": prompt})
#         if completion is not None:
#             conversation.append({"role": "assistant", "content": completion})
#     return conversation


# def set_defaults(model_name):
#     config = MODEL_CONFIGS[model_name]
#     return (
#         model_name,
#         gr.update(choices=config["concepts"], value=config["concepts"][0]),
#         gr.update(
#             minimum=config["guidance_interval"][0],
#             maximum=config["guidance_interval"][1],
#             value=config["default_guidance_scale"],
#         ),
#         gr.update(value=config["min_guidance_layer"]),
#         gr.update(value=config["max_guidance_layer"]),
#     )


# def add_user_prompt(user_message, history):
#     if history is None:
#         history = []
#     history.append([user_message, None])
#     return history


# @torch.no_grad()
# def generate_completion(
#     history,
#     model_name,
#     concept,
#     guidance_scale=2.5,
#     min_guidance_layer=16,
#     max_guidance_layer=32,
#     temperature=0.0,
#     repetition_penalty=1.2,
#     length_penalty=1.2,
# ):
#     start_time = time.time()
#     logger.info(
#         f" --- Starting completion ({model_name}, {concept=}, {guidance_scale=},"
#         f" {min_guidance_layer=}, {temperature=})"
#     )
#     logger.info(" User: " + repr(history[-1][0]))

#     torch.cuda.empty_cache()
#     # load the model
#     config = MODEL_CONFIGS[model_name]
#     if gr.NO_RELOAD:
#         model, tokenizer = load_model()
#     # if not config["load_in_8bit"]:
#     #     model.to(device, non_blocking=True)

#     steering_vector = load_steering_vectors()
#     # guidance_layers = list(range(int(min_guidance_layer) - 1, int(max_guidance_layer)))
#     apply_steering(
#         model,
#         steering_vector,
#         scale=guidance_scale,
#     )

#     pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         # device=(device if not config["load_in_8bit"] else None),
#     )

#     conversation = history_to_conversation(history)
#     streamer = TextIteratorStreamer(
#         tokenizer, skip_prompt=True, skip_special_tokens=True
#     )

#     generation_kwargs = dict(
#         max_new_tokens=512,
#         repetition_penalty=float(repetition_penalty),
#         length_penalty=length_penalty,
#         streamer=streamer,
#         temperature=temperature,
#         do_sample=(temperature > 0),
#     )
#     thread = Thread(
#         target=pipe, args=(conversation,), kwargs=generation_kwargs, daemon=True
#     )
#     thread.start()

#     history[-1][1] = ""
#     for token in streamer:
#         history[-1][1] += token
#         yield history
#     logger.info(" Assistant: " + repr(history[-1][1]))

#     time_taken = time.time() - start_time
#     logger.info(f" --- Completed (took {time_taken:.1f}s)")
#     return history


# class ConceptGuidanceUI:
#     def __init__(self):
#         model_names = list(MODEL_CONFIGS.keys())
#         default_model = model_names[0]
#         default_config = MODEL_CONFIGS[default_model]
#         default_concepts = default_config["concepts"]
#         default_concept = default_config["default_concept"]

#         saved_input = gr.State("")

#         with gr.Row(elem_id="concept-guidance-container"):
#             with gr.Column(scale=1, min_width=256):
#                 model_dropdown = gr.Dropdown(
#                     model_names, value=default_model, label="Model", visible=False
#                 )
#                 concept_dropdown = gr.Dropdown(
#                     default_concepts,
#                     value=default_concept,
#                     label="Concept",
#                     visible=False,
#                 )
#                 guidance_scale = gr.Slider(
#                     *default_config["guidance_interval"],
#                     value=2.5,
#                     label="Guidance Scale",
#                 )
#                 min_guidance_layer = gr.Slider(
#                     1.0, 32.0, value=16.0, step=1.0, label="First Guidance Layer"
#                 )
#                 max_guidance_layer = gr.Slider(
#                     1.0, 32.0, value=32.0, step=1.0, label="Last Guidance Layer"
#                 )
#                 temperature = gr.Slider(
#                     0.0, 1.0, value=0.0, step=0.01, label="Temperature"
#                 )
#                 repetition_penalty = gr.Slider(
#                     1.0,
#                     5.0,
#                     value=1.2,
#                     step=0.01,
#                     label="Repetition Penalty",
#                 )
#                 length_penalty = gr.Slider(
#                     -10.0, 10.0, value=1.2, step=0.01, label="Length Penalty"
#                 )

#             with gr.Column(scale=3, min_width=512):
#                 chatbot = gr.Chatbot(scale=1, height=800)

#                 with gr.Row():
#                     self.retry_btn = gr.Button("🔄 Retry", size="sm")
#                     self.undo_btn = gr.Button("↩️ Undo", size="sm")
#                     self.clear_btn = gr.Button("🗑️ Clear", size="sm")

#                 with gr.Group():
#                     with gr.Row():
#                         prompt_field = gr.Textbox(
#                             placeholder="Type a message...",
#                             show_label=False,
#                             label="Message",
#                             scale=7,
#                             container=False,
#                         )
#                         self.submit_btn = gr.Button(
#                             "Submit", variant="primary", scale=1, min_width=150
#                         )
#                         self.stop_btn = gr.Button(
#                             "Stop",
#                             variant="secondary",
#                             scale=1,
#                             min_width=150,
#                             visible=False,
#                         )

#         generation_args = [
#             model_dropdown,
#             concept_dropdown,
#             guidance_scale,
#             min_guidance_layer,
#             max_guidance_layer,
#             temperature,
#             repetition_penalty,
#             length_penalty,
#         ]

#         model_dropdown.change(
#             set_defaults,
#             [model_dropdown],
#             [
#                 model_dropdown,
#                 concept_dropdown,
#                 guidance_scale,
#                 min_guidance_layer,
#                 max_guidance_layer,
#             ],
#             queue=False,
#         )

#         submit_triggers = [prompt_field.submit, self.submit_btn.click]
#         submit_event = (
#             gr.on(
#                 submit_triggers,
#                 self.clear_and_save_input,
#                 [prompt_field],
#                 [prompt_field, saved_input],
#                 queue=False,
#             )
#             .then(add_user_prompt, [saved_input, chatbot], [chatbot], queue=False)
#             .then(
#                 generate_completion,
#                 [chatbot] + generation_args,
#                 [chatbot],
#                 concurrency_limit=1,
#             )
#         )
#         self.setup_stop_events(submit_triggers, submit_event)

#         retry_triggers = [self.retry_btn.click]
#         retry_event = (
#             gr.on(
#                 retry_triggers,
#                 self.delete_prev_message,
#                 [chatbot],
#                 [chatbot, saved_input],
#                 queue=False,
#             )
#             .then(add_user_prompt, [saved_input, chatbot], [chatbot], queue=False)
#             .then(
#                 generate_completion,
#                 [chatbot] + generation_args,
#                 [chatbot],
#                 concurrency_limit=1,
#             )
#         )
#         self.setup_stop_events(retry_triggers, retry_event)

#         self.undo_btn.click(
#             self.delete_prev_message, [chatbot], [chatbot, saved_input], queue=False
#         ).then(lambda x: x, [saved_input], [prompt_field])
#         self.clear_btn.click(
#             lambda: [None, None], None, [chatbot, saved_input], queue=False
#         )

#     def clear_and_save_input(self, message):
#         return "", message

#     def delete_prev_message(self, history):
#         message, _ = history.pop()
#         return history, message or ""

#     def setup_stop_events(self, event_triggers, event_to_cancel):
#         if self.submit_btn:
#             for event_trigger in event_triggers:
#                 event_trigger(
#                     lambda: (
#                         gr.Button(visible=False),
#                         gr.Button(visible=True),
#                     ),
#                     None,
#                     [self.submit_btn, self.stop_btn],
#                     show_api=False,
#                     queue=False,
#                 )
#             event_to_cancel.then(
#                 lambda: (gr.Button(visible=True), gr.Button(visible=False)),
#                 None,
#                 [self.submit_btn, self.stop_btn],
#                 show_api=False,
#                 queue=False,
#             )

#         self.stop_btn.click(
#             None,
#             None,
#             None,
#             cancels=event_to_cancel,
#             show_api=False,
#         )


# css = """
# #concept-guidance-container {
#     flex-grow: 1;
# }
# """.strip()

# with gr.Blocks(
#     title="Concept Guidance", fill_height=True, css=css, theme="lone17/kotaemon"
# ) as demo:
#     ConceptGuidanceUI()

# demo.queue()
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--share", action="store_true")
#     args = parser.parse_args()
#     demo.launch(share=args.share, server_name="0.0.0.0", server_port=8999)
