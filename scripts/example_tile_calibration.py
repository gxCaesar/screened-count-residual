#!/usr/bin/env python3
"""Minimal CPU worked example: fit one Screened Count Residual head, reload, and verify zero-DC gauge."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from s2 import screened_universal_head as suh


def main() -> int:
    torch.manual_seed(42)
    n_tiles, h, w = 8, 64, 64
    he = torch.rand(n_tiles, 3, h, w)
    # Synthetic sparse biomarker spots attenuated in the upstream prediction
    target_f = torch.zeros(n_tiles, 1, h, w)
    target_f[:, :, 16:28, 16:28] = 0.65
    target_f[:, :, 40:50, 38:48] = 0.55
    donor_f = (0.18 * target_f + 0.02 * torch.rand_like(target_f)).clamp(0.0, 1.0)
    struct_f = (0.5 * he[:, 0:1] + 0.5 * he[:, 2:3]).clamp(0.0, 1.0)
    mean_f = 0.5 * (donor_f + struct_f)

    common12 = suh.build_universal_common12(he, donor_f, struct_f, mean_f)
    donor_u8 = torch.round(donor_f[:, 0] * 255.0).to(torch.uint8)
    target_u8 = torch.round(target_f[:, 0] * 255.0).to(torch.uint8)

    model, meta = suh.fit_universal_marker(
        arm="S_count",
        seed=202610270,
        common_train=common12,
        donor_u8_train=donor_u8,
        target_u8_train=target_u8,
        cutoff=23,
        device=torch.device("cpu"),
        epochs=2,
        batch_size=4,
    )
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "example_sc_head.pt"
        torch.save(model.state_dict(), ckpt)
        reloaded = suh.build_coordinate_screened_model(202610270, torch.device("cpu"))
        reloaded.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        reloaded.eval()
        with torch.no_grad():
            residual, diag = suh.fast_sc_forward(reloaded, common12[:2], inference=True)
    dc_max = float(residual.mean(dim=(-2, -1)).abs().max().item())
    cg_rel = float(diag["relative_residual_max"])
    assert dc_max < 1e-10, f"Pre-clipping zero-DC violated: {dc_max}"
    assert cg_rel < 1e-10, f"CG relative residual violated: {cg_rel}"
    print(f"Example tile calibration passed: dc_max={dc_max:.2e}, cg_rel={cg_rel:.2e}, arm={meta['arm']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
