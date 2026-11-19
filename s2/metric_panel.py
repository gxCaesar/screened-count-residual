"""Metric panel for S2: which per-marker metrics can be gamed by blanking or by blurring?

Per val image and marker (channels R, G, B = panCK, CD3, DAPI):
  ssim, psnr       as s2_scoring (DTR's SSIM._ssim, uint8 PSNR)
  r_drop, r0       Pearson R; r_drop is NaN for a constant prediction (dropped by nanmean, as in the
                   charter's guard), r0 counts it as 0
  f1_tol           positive-region F1 with a 5-pixel tolerance: positives are pixels above the marker's
                   threshold (Otsu on pooled train-label pixels: panCK 22, CD3 30, DAPI 18, fixed in
                   FIXED_THR); precision is the share of predicted positives within 5 px of a true
                   positive, recall the converse. Per image, both empty gives 1 and exactly one empty
                   gives 0, which makes the per-image mean jump on single false-positive pixels in
                   label-empty patches
  f1_pool          THE PRIMARY METRIC (charter amendment 2026-10-04): the same tolerance F1 from counts
                   pooled over all val images per marker (sum of matched predicted positives over sum
                   of predicted positives, and likewise for recall), then averaged over markers.
                   Empty patches contribute their false-positive pixels and nothing else
  frac_pred/gt     positive-area fraction, for the across-image Pearson in the summary (a patch-level
                   quantification readout: T-cell density, epithelial fraction, nuclear density)

Arms: each checkpoint passed with --ckpt NAME=PATH, plus, from the first checkpoint: the same
output with one marker blanked (one arm per marker), the output blurred (sigma 2 and 4), and
the constant image.

Run from the DTR checkout:
  python metric_panel.py --data_root HEMIT --out OUT.json --ckpt released=W [--ckpt ctrl_s0=W ...]
"""
import argparse
import json
import os
import sys

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())
import s2_scoring  # noqa: E402
from metrics import SSIM  # noqa: E402

TRUE = ["panCK", "CD3", "DAPI"]
TOL = 5
FIXED_THR = {"panCK": 22.0, "CD3": 30.0, "DAPI": 18.0}  # Otsu on train labels, metric_panel_v1 2026-10-04


def otsu(values):
    v = np.clip(values, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(v.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t)


def pearson(x, y):
    x = x.astype(np.float64).ravel()
    y = y.astype(np.float64).ravel()
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def tol_counts(pm, gm, tol=TOL):
    """(predicted positives, of them within tol of a true positive, true positives, of them within tol
    of a predicted positive)."""
    n_p, n_g = int(pm.sum()), int(gm.sum())
    tp_p = int((cv2.distanceTransform((~gm).astype(np.uint8), cv2.DIST_L2, 5)[pm] <= tol).sum()) if n_p and n_g else 0
    tp_g = int((cv2.distanceTransform((~pm).astype(np.uint8), cv2.DIST_L2, 5)[gm] <= tol).sum()) if n_p and n_g else 0
    return n_p, tp_p, n_g, tp_g


def f1_from_counts(n_p, tp_p, n_g, tp_g):
    if n_p == 0 and n_g == 0:
        return 1.0
    if n_p == 0 or n_g == 0:
        return 0.0
    prec, rec = tp_p / n_p, tp_g / n_g
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def f1_tol(pm, gm, tol=TOL):
    return f1_from_counts(*tol_counts(pm, gm, tol))


def score(pred, gt, thr):
    out = {}
    for c, m in enumerate(TRUE):
        p, g = pred[..., c], gt[..., c]
        mse = np.mean((p.astype(np.float64) - g.astype(np.float64)) ** 2)
        out[f"{m}_psnr"] = float(20 * np.log10(255.0 / (np.sqrt(mse) + 1e-8)))
        out[f"{m}_ssim"] = float(SSIM._ssim(p, g))
        r = pearson(p, g)
        out[f"{m}_r_drop"] = r
        out[f"{m}_r0"] = 0.0 if np.isnan(r) else r
        pm, gm = p > thr[m], g > thr[m]
        n_p, tp_p, n_g, tp_g = tol_counts(pm, gm)
        out[f"{m}_f1_tol"] = f1_from_counts(n_p, tp_p, n_g, tp_g)
        out[f"{m}_np"], out[f"{m}_tpp"], out[f"{m}_ng"], out[f"{m}_tpg"] = n_p, tp_p, n_g, tp_g
        out[f"{m}_frac_pred"] = float(pm.mean())
        out[f"{m}_frac_gt"] = float(gm.mean())
    return out


def self_test():
    rng = np.random.default_rng(0)
    g = np.zeros((96, 96, 3), np.uint8)
    for _ in range(12):
        y, x = rng.integers(8, 88, 2)
        cv2.circle(g, (int(x), int(y)), 4, (200, 200, 200), -1)
    thr = {m: 100.0 for m in TRUE}
    s = score(g, g, thr)
    assert all(abs(s[f"{m}_f1_tol"] - 1) < 1e-9 and abs(s[f"{m}_r0"] - 1) < 1e-9 for m in TRUE)
    b = score(np.zeros_like(g), g, thr)
    assert all(b[f"{m}_f1_tol"] == 0 and b[f"{m}_r0"] == 0 and np.isnan(b[f"{m}_r_drop"]) for m in TRUE)
    sh = np.roll(g, 3, axis=1)
    s3 = score(sh, g, thr)
    assert all(abs(s3[f"{m}_f1_tol"] - 1) < 1e-9 for m in TRUE), "a 3 px shift is inside the 5 px tolerance"
    s9 = score(np.roll(g, 12, axis=1), g, thr)
    assert all(s9[f"{m}_f1_tol"] < 0.6 for m in TRUE)
    e = score(np.zeros_like(g), np.zeros_like(g), thr)
    assert all(e[f"{m}_f1_tol"] == 1 for m in TRUE)
    # pooled: an empty-label image with one false-positive pixel costs one pixel, not the whole image
    one = np.zeros_like(g); one[10, 10] = 255
    rows = [score(g, g, thr), score(one, np.zeros_like(g), thr)]
    pooled = pooled_f1(rows)
    per_image = np.mean([r["CD3_f1_tol"] for r in rows])
    assert pooled["CD3"] > 0.999 and abs(per_image - 0.5) < 1e-9, (pooled, per_image)
    assert pooled_f1([score(np.zeros_like(g), g, thr)])["CD3"] == 0.0
    return True


def pooled_f1(rows, prefix=""):
    out = {}
    for m in TRUE:
        n_p = sum(r[f"{prefix}{m}_np"] for r in rows)
        tp_p = sum(r[f"{prefix}{m}_tpp"] for r in rows)
        n_g = sum(r[f"{prefix}{m}_ng"] for r in rows)
        tp_g = sum(r[f"{prefix}{m}_tpg"] for r in rows)
        out[m] = f1_from_counts(n_p, tp_p, n_g, tp_g)
    out["avg"] = float(np.mean([out[m] for m in TRUE]))
    return out


def summarise(rows, arms):
    """Per-arm summary from rows keyed '<arm>__<marker>_<quantity>' (shared with dgr_ft.py)."""
    summ = {}
    for arm in arms:
        s = {}
        for m in TRUE:
            for q in ("psnr", "ssim", "r_drop", "r0", "f1_tol"):
                s[f"{m}_{q}"] = float(np.nanmean([r[f"{arm}__{m}_{q}"] for r in rows]))
            fp = np.array([r[f"{arm}__{m}_frac_pred"] for r in rows])
            fg = np.array([r[f"{arm}__{m}_frac_gt"] for r in rows])
            s[f"{m}_frac_corr"] = pearson(fp, fg)
            s[f"{m}_frac_corr0"] = 0.0 if np.isnan(s[f"{m}_frac_corr"]) else s[f"{m}_frac_corr"]
        for q in ("psnr", "ssim", "r_drop", "r0", "f1_tol", "frac_corr0"):
            s[f"avg_{q}"] = float(np.mean([s[f"{m}_{q}"] for m in TRUE]))
        pf = pooled_f1(rows, prefix=f"{arm}__")
        for m in TRUE:
            s[f"{m}_f1_pool"] = pf[m]
        s["avg_f1_pool"] = pf["avg"]
        summ[arm] = s
    return summ


def load_gen(path, dev):
    import Attention_GAN
    netG = Attention_GAN.Generator(n_channels=64, in_channels=3, batch_norm=False, out_channels=3,
                                   padding=1, pooling_mode="maxpool").to(dev)
    sd = {k.replace("module.", "", 1): v for k, v in torch.load(path, map_location="cpu").items()}
    model = netG
    if "fml_cfg" in sd:
        import s2_components
        return s2_components.FMLGen.from_state(netG, sd, dev).eval()
    if any(k.startswith("inner.") for k in sd):
        import s2_components
        model = s2_components.ColourPriorWrap(netG, sd["W"].tolist()).to(dev)
    model.load_state_dict(sd, strict=True)
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--n_thr_images", type=int, default=100)
    ap.add_argument("--no_aux", action="store_true", help="skip the blank / blur / constant arms")
    ap.add_argument("--recompute_thr", action="store_true", help="recompute Otsu thresholds and assert they equal FIXED_THR")
    a = ap.parse_args()
    assert not os.path.exists(a.out)
    assert self_test()

    thr = dict(FIXED_THR)
    if a.recompute_thr:
        thr = otsu_thresholds(a.data_root, a.n_thr_images)
        assert thr == FIXED_THR, thr
    print("thresholds:", thr, flush=True)
    run_panel(a, thr)


def otsu_thresholds(data_root, n_thr_images):
    # Otsu per marker on pooled pixels of train labels (centre crops, every k-th image)
    tr_dir = os.path.join(data_root, "train", "label")
    names = sorted(os.listdir(tr_dir))
    step = max(1, len(names) // n_thr_images)
    pix = {m: [] for m in TRUE}
    rng = np.random.default_rng(0)
    for n in names[::step][: n_thr_images]:
        y = np.asarray(Image.open(os.path.join(tr_dir, n)).convert("RGB"))[256:768, 256:768]
        idx = rng.choice(512 * 512, 4000, replace=False)
        for c, m in enumerate(TRUE):
            pix[m].append(y[..., c].ravel()[idx])
    return {m: otsu(np.concatenate(pix[m])) for m in TRUE}


def run_panel(a, thr):
    from batch_utils import UNI
    dev = torch.device("cuda")
    ckpts = [s.split("=", 1) for s in a.ckpt]
    models = {n: load_gen(p, dev) for n, p in ckpts}
    first = ckpts[0][0]
    val = UNI("hemit", a.data_root, 512, "val", 0)
    const = None
    rows = []
    with torch.no_grad():
        for i in range(len(val)):
            it = val[i]
            gt = s2_scoring.to_u8(it["gt"])
            preds = {n: s2_scoring.to_u8(m(it["input"][None].to(dev))[0]) for n, m in models.items()}
            if not a.no_aux:
                base = preds[first]
                for c, m in enumerate(TRUE):
                    bl = base.copy()
                    bl[..., c] = 0
                    preds[f"{first}_blank_{m}"] = bl
                for s in (2, 4):
                    preds[f"{first}_blur{s}"] = cv2.GaussianBlur(base, (0, 0), s)
                if const is None:
                    const = np.broadcast_to(np.array([10, 4, 10], np.uint8), gt.shape).copy()  # step0 constant
                preds["constant"] = const
            row = {"name": it["name"]}
            for arm, pr in preds.items():
                for k, v in score(pr, gt, thr).items():
                    row[f"{arm}__{k}"] = v
            rows.append(row)
            if i % 100 == 0:
                print(i, flush=True)

    arms = [k.split("__")[0] for k in rows[0] if k.endswith(f"__{TRUE[0]}_ssim")]
    summ = summarise(rows, arms)
    json.dump({"thresholds": thr, "tol_px": TOL, "summary": summ, "rows": rows}, open(a.out, "w"))
    cols = ["avg_f1_pool", "avg_f1_tol", "avg_ssim", "avg_psnr", "avg_r0", "avg_frac_corr0",
            "panCK_f1_pool", "CD3_f1_pool", "DAPI_f1_pool"]
    print(f"{'arm':22s}" + "".join(f"{c:>15s}" for c in cols))
    for arm in arms:
        print(f"{arm:22s}" + "".join(f"{summ[arm][c]:15.4f}" for c in cols))


if __name__ == "__main__":
    main()
