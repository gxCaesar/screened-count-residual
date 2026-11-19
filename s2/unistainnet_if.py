"""Source-near HEMIT adaptation of the separately pinned UNIStainNet trainer.

Place the pinned integration source root on sys.path before importing this
module. Its IHC trainer is not modified. Validation LPIPS/SSIM are source
selection diagnostics; final IF scores use the project's common evaluator.
"""
import torch
import torch.nn.functional as F

from src.models.trainer import UNIStainNetTrainer


RECIPE_VERSION = 'unistainnet-hemit-source-near-v1'
STAIN_WEIGHTS = ('dab_intensity_weight', 'dab_contrast_weight',
                 'dab_sharpness_weight', 'bg_white_weight')


class UNIStainNetIFTrainer(UNIStainNetTrainer):
    """Preserve upstream training with no DAB objective or grade inputs."""

    def __init__(self, **kwargs):
        for name in STAIN_WEIGHTS:
            if kwargs.get(name, 0.0) != 0.0:
                raise ValueError(f'{name} must be zero for the IF recipe')
            kwargs[name] = 0.0
        for name, expected in (('null_class', 4), ('num_classes', 5), ('image_size', 512)):
            if kwargs.get(name, expected) != expected:
                raise ValueError(f'{name} must be {expected} for the IF recipe')
            kwargs[name] = expected
        if kwargs.get('label_names') not in (None, []):
            raise ValueError('IF recipe does not use HER2 grade names')
        kwargs['label_names'] = None
        super().__init__(**kwargs)

    def _require_null_labels(self, labels):
        if (not torch.is_tensor(labels) or labels.ndim != 1
                or labels.dtype != torch.long or labels.numel() == 0
                or not bool(torch.all(labels == self.null_class))):
            raise ValueError('IF recipe requires nonempty constant null labels (int64, value 4)')

    def training_step(self, batch, batch_idx):
        self._require_null_labels(batch[3])
        return super().training_step(batch, batch_idx)

    def on_validation_epoch_start(self):
        # No random or class-labelled grids in this manifest-only IF path.
        pass

    def validation_step(self, batch, batch_idx):
        he, target, uni_or_crops, labels, _ = batch
        self._require_null_labels(labels)
        uni = (self._extract_uni_from_sub_crops(uni_or_crops)
               if self._uni_extract_on_the_fly else uni_or_crops)
        if self.hparams.disable_uni:
            uni = torch.zeros_like(uni)
        with torch.no_grad():
            generated = self.generator_ema(he, uni, labels)
        size = self.hparams.image_size // 4
        pred_lpips = F.interpolate(generated, size=size, mode='bilinear', align_corners=False)
        true_lpips = F.interpolate(target, size=size, mode='bilinear', align_corners=False)
        lpips_val = self.lpips_fn(pred_lpips, true_lpips).mean()
        from torchmetrics.functional.image import structural_similarity_index_measure
        ssim_val = structural_similarity_index_measure(
            ((generated + 1) / 2).clamp(0, 1),
            ((target + 1) / 2).clamp(0, 1), data_range=1.0)
        self.log('val/lpips', lpips_val, prog_bar=True, sync_dist=True, batch_size=len(he))
        self.log('val/ssim', ssim_val, prog_bar=True, sync_dist=True, batch_size=len(he))

    def on_validation_epoch_end(self):
        pass

    @torch.no_grad()
    def generate(self, he_images, uni_features, labels, num_inference_steps=None,
                 guidance_scale=1.0, seed=None):
        self._require_null_labels(labels)
        return super().generate(he_images, uni_features, labels,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale, seed=seed)
