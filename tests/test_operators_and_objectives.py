"""Unit tests for gauge-fixed Screened Count Residual operators and clinical survival helpers."""
from __future__ import annotations

import unittest
import numpy as np
import torch

from s2 import clinical_cell_survival as ccs
from s2 import query_relation_reconstruction as qrr
from s2 import run_her2match_dab_campaign as rhdc
from s2 import screened_universal_head as suh


class TestScreenedUniversalOperators(unittest.TestCase):
    def test_zero_dc_gauge_invariance_and_solver_accuracy(self) -> None:
        torch.manual_seed(7)
        u = torch.randn(2, 1, 64, 64, dtype=torch.float64)
        lu = qrr.laplacian(u)
        r = qrr.screened_solve(lu, inference=True)
        dc_mean = float(r.mean(dim=(-2, -1)).abs().max().item())
        res_err = float(((r + qrr.laplacian(r)) - lu).abs().max().item())
        self.assertLess(dc_mean, 1e-10)
        self.assertLess(res_err, 1e-9)

        r_shifted = qrr.screened_solve(qrr.laplacian(u + 13.5), inference=True)
        self.assertLess(float((r - r_shifted).abs().max().item()), 1e-9)

    def test_detached_count_objective_isolated_fp_penalty_and_tp_boost(self) -> None:
        # Unclamped [0, 1] field verifying non-saturated gradients: isolated FP outside halo
        # receives positive loss gradient (pushing intensity down), while TP inside target
        # receives negative loss gradient (pushing intensity up).
        z = torch.full((1, 1, 32, 32), 0.08, dtype=torch.float32, requires_grad=True)
        targ = torch.zeros((1, 1, 32, 32), dtype=torch.uint8)
        targ[:, :, 4:10, 4:10] = 60  # true positive block far from (26, 26)
        prep = suh.prepare_channel_counts(z, targ, cutoff=23)
        kuv = suh.deployment_cutoff_densities(prep, cutoff=23)
        mean_kuv = kuv.mean(dim=0)
        mean_g = float(prep["G"].mean().item())
        weights, f_val = suh.adjoint_from_single_row(mean_kuv.detach(), mean_g)
        self.assertGreater(float(weights[0].item()), 0.0)  # +2 u v^2 / (3 d^2) > 0 penalizes K
        loss = (weights * mean_kuv).sum()
        loss.backward()
        self.assertTrue(np.isfinite(f_val))
        self.assertIsNotNone(z.grad)
        self.assertTrue(torch.isfinite(z.grad).all().item())
        # Isolated false-positive site at (26, 26) outside 5-px halo of [4:10, 4:10]
        self.assertGreater(float(z.grad[0, 0, 26, 26].item()), 0.0)
        # True-positive interior site at (6, 6)
        self.assertLess(float(z.grad[0, 0, 6, 6].item()), 0.0)

    def test_universal_common12_features_and_head_forward(self) -> None:
        he = torch.rand(2, 3, 64, 64)
        bc = torch.rand(2, 1, 64, 64)
        bs = torch.rand(2, 1, 64, 64)
        bm = 0.5 * (bc + bs)
        c12 = suh.build_universal_common12(he, bc, bs, bm)
        self.assertEqual(tuple(c12.shape), (2, 12, 64, 64))
        model = suh.build_coordinate_screened_model(202610270, torch.device("cpu"))
        residual, diag = suh.fast_sc_forward(model, c12, inference=True)
        self.assertEqual(tuple(residual.shape), (2, 1, 64, 64))
        self.assertLess(float(residual.mean(dim=(-2, -1)).abs().max().item()), 1e-10)
        self.assertIn("relative_residual_max", diag)

    def test_ruifrok_johnston_dab_channel_and_strict_cutoff_boundary(self) -> None:
        # Pure brown DAB-like patch vs purple-blue Hematoxylin-like patch on white background
        rgb = np.full((32, 32, 3), 245, dtype=np.uint8)
        rgb[4:12, 4:12] = np.array([120, 75, 45], dtype=np.uint8)    # brown DAB
        rgb[20:28, 20:28] = np.array([75, 70, 145], dtype=np.uint8)  # purple Hematoxylin
        stains = rhdc.separate_stains(rgb.astype(np.float64) / 255.0, rhdc.hdx_from_rgb)
        hema_od = stains[..., 0]
        dab_od = stains[..., 1]
        self.assertGreater(float(dab_od[8, 8]), float(dab_od[24, 24]))
        self.assertGreater(float(hema_od[24, 24]), float(hema_od[8, 8]))
        # Verify strict byte positivity > 23 (23 is negative, 24 is positive)
        t_u8 = torch.tensor([[[[23, 24]]]], dtype=torch.uint8)
        prep = suh.prepare_channel_counts(torch.full((1, 1, 1, 2), 0.1), t_u8, cutoff=23)
        self.assertFalse(bool(prep["g"][0, 0, 0, 0].item()))
        self.assertTrue(bool(prep["g"][0, 0, 0, 1].item()))

    def test_duplicate_free_harrell_c_index_and_survival_bootstrap(self) -> None:
        days = np.array([120, 180, 240, 410, 620, 800, 950, 1100], dtype=np.float64)
        ev = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int32)
        score = np.array([0.4, 0.8, 1.1, 1.5, 2.6, 3.0, 3.4, 3.8], dtype=np.float64)
        # Verify self-pairs with identical unit_ids are excluded
        dup_days = np.array([100.0, 100.0, 200.0])
        dup_ev = np.array([1, 1, 0], dtype=np.int32)
        dup_sc = np.array([1.0, 1.0, 2.0])
        c_no_self = ccs.harrell_c_index(dup_days, dup_ev, dup_sc, unit_ids=np.array([0, 0, 1]))
        self.assertAlmostEqual(c_no_self, 1.0, places=6)
        surv = ccs.evaluate_pfs_survival(pfs_days=days, recurrence_events=ev, protective_score=score, dichotomy_cutoff=2.0, n_bootstrap=200)
        self.assertGreater(float(surv["c_index"]), 0.85)

    def test_plan2_ablation_runner_preallocates_zero_mean_arm(self) -> None:
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "s2" / "run_plan2_ablations.py").read_text()
        self.assertGreaterEqual(src.count('"E_count", "E_count_zm", "Unsharp"'), 2)


if __name__ == "__main__":
    unittest.main()
