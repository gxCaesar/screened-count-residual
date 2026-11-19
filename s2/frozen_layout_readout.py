"""Frozen affine marker fractions; coefficients are buffers, never logits."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def validate_weights(weight, bias):
    for name, value, shape in (('weight', weight, (3, 1536, 1, 1)),
                                ('bias', bias, (3,))):
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f'{name} must be a tensor of shape {shape}')
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f'{name} must contain finite floating-point values')
        if not torch.isfinite(value.float()).all():
            raise ValueError(f'{name} overflows float32')


class FrozenLayoutReadout(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        validate_weights(weight, bias)
        if weight.device != bias.device:
            raise ValueError('weight and bias must share a device')
        self.register_buffer('weight', weight.detach().float().clone())
        self.register_buffer('bias', bias.detach().float().clone())

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Validate both tensors before either existing buffer can be mutated,
        # including recursive loading inside renderer/EMA/SWA modules.
        try:
            validate_weights(state_dict[prefix + 'weight'], state_dict[prefix + 'bias'])
        except (KeyError, ValueError) as error:
            raise RuntimeError(f'invalid frozen readout state: {error}') from error
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, tokens):
        if tokens.ndim != 4 or tokens.shape[1] != 1536:
            raise ValueError('tokens must have shape [B,1536,H,W]')
        with torch.autocast(tokens.device.type, enabled=False), torch.backends.cudnn.flags(allow_tf32=False):
            return F.conv2d(tokens.float(), self.weight.float(), self.bias.float()).clamp(0, 1)


def load_readout_payload(path):
    path = Path(path).absolute()
    if ('..' in path.parts or path.resolve(strict=True) != path or not path.is_file() or
            any(part.lower() == 'test' or 'score' in part.lower() for part in path.parts)):
        raise ValueError('readout must be a canonical regular file outside test/score paths')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {'weight', 'bias'}:
        raise ValueError('payload must contain only weight and bias tensors')
    validate_weights(payload['weight'], payload['bias'])
    return payload


def load_readout(path):
    """Load a canonical tensor-only payload into a CPU buffer-only readout."""
    return FrozenLayoutReadout(**load_readout_payload(path))
