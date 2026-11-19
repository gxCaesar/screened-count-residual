"""Matched UNI2-h adaptation of the source-near UNIStainNet IF trainer.

This is a named backbone/input-view substitution, not the official UNI recipe.
The frozen encoder and its preprocessing reuse the current FML implementation.
It stays in that implementation's process cache, outside renderer checkpoints.
"""
import os

import torch

import s2_components as components
from unistainnet_if import UNIStainNetIFTrainer


RECIPE_VERSION = 'unistainnet-hemit-matched-uni2h-v1'
FEATURE_VERSION = 'fml-uni2h-448-bilinear-aa-32grid-v1'
_TOKEN_VIEWS = {}


@torch.no_grad()
def matched_tokens(he):
    if (not torch.is_tensor(he) or he.ndim != 4
            or tuple(he.shape[1:]) != (3, 512, 512) or he.numel() == 0
            or not he.is_floating_point() or not bool(torch.isfinite(he).all())
            or bool((he < -1).any() or (he > 1).any())):
        raise ValueError('matched UNI2-h requires finite H&E [B,3,512,512] in [-1,1]')
    # The launcher also sets this before imports. Never acquire new weights.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    fm = components._fm(he.device, False)
    if fm.training or any(p.requires_grad for p in fm.parameters()):
        raise ValueError('matched UNI2-h must be frozen and in eval mode')
    if fm.num_features != 1536:
        raise ValueError('matched UNI2-h requires 1536 features')
    key = str(he.device)
    if key not in _TOKEN_VIEWS:
        _TOKEN_VIEWS[key] = components.FMLGen(
            torch.nn.Identity(), he.device, layout=False, tokens=False).to(he.device)
    grid = _TOKEN_VIEWS[key].tokens_of(he)
    if tuple(grid.shape) != (len(he), 1536, 32, 32) or not bool(torch.isfinite(grid).all()):
        raise ValueError('matched UNI2-h must produce finite [B,1536,32,32] features')
    return grid.flatten(2).transpose(1, 2).contiguous()


class MatchedUNIStainNetIFTrainer(UNIStainNetIFTrainer):
    def __init__(self, matched_feature_recipe=FEATURE_VERSION, **kwargs):
        if matched_feature_recipe != FEATURE_VERSION:
            raise ValueError('unknown matched UNI2-h feature recipe')
        for name, expected in (('uni_dim', 1536), ('uni_spatial_size', 32),
                               ('uni_spatial_pool_size', 32),
                               ('extract_uni_on_the_fly', False), ('disable_uni', False)):
            if kwargs.get(name, expected) != expected:
                raise ValueError(f'{name} must be {expected} for matched UNI2-h')
            kwargs[name] = expected
        super().__init__(**kwargs)
        self.save_hyperparameters({'matched_feature_recipe': matched_feature_recipe})

    def _load_uni_model(self):
        raise RuntimeError('the original UNI encoder is disabled in the matched UNI2-h recipe')

    def _extract_uni_from_sub_crops(self, uni_sub_crops):
        raise RuntimeError('matched UNI2-h uses the final H&E tensor, not UNI sub-crops')

    def features_from_he(self, he):
        return matched_tokens(he)

    def _feature_batch(self, batch):
        he, target, empty_view, labels, names = batch
        self._require_null_labels(labels)
        if (not torch.is_tensor(empty_view) or empty_view.ndim != 2
                or tuple(empty_view.shape) != (len(he), 0)):
            raise ValueError('matched UNI2-h training requires loader fm_view=none')
        return he, target, self.features_from_he(he), labels, names

    def training_step(self, batch, batch_idx):
        return super().training_step(self._feature_batch(batch), batch_idx)

    def validation_step(self, batch, batch_idx):
        return super().validation_step(self._feature_batch(batch), batch_idx)
