import torch
from torch import nn
from transformers import AutoModelForCausalLM


class TransformerDecoderBlockWithForcedDirection(nn.Module):
    def __init__(self, decoder_layer, forced_direction: torch.Tensor = None):
        super().__init__()

        self.decoder_layer = decoder_layer

        scale = nn.Parameter(
            torch.tensor(0.0, dtype=decoder_layer.dtype), requires_grad=True
        )
        scale.to(decoder_layer.device)
        self.scale = scale

        if not isinstance(forced_direction, torch.Tensor):
            forced_direction = torch.tensor(forced_direction)
        forced_direction = forced_direction.to(
            decoder_layer.device, dtype=decoder_layer.dtype
        )
        self.forced_direction = forced_direction

    def forward(self, x, **kwargs):
        x = self.decoder_layer(x, **kwargs)
        x += self.scale * self.forced_direction

        return x


model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
for param in model.parameters():
    param.requires_grad = False
