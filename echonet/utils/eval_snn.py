"""
eval_snn.py — Full CNN vs SNN Evaluation on EchoNet-Dynamic validation set.

Outputs:
  - MAE, RMSE, R² for both CNN and SNN
  - Side-by-side comparison table
  - Saved predictions to snn_eval_results.pth

Run:
  python -m echonet.utils.eval_snn \
      --data_dir ./EchoNet-Dynamic \
      --weights   ./output/video/r2plus1d_18_32_2_pretrained/best.pt
"""

import argparse
import copy
import math

import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import tqdm

from echonet.utils.video import video_run_internal
from echonet.snn.rnl import RateNormalizer, save_max_activations
from echonet.snn.conversion import fold_batchnorm, convert_to_snn
from echonet.snn.spiking_model import SpikingModel
from echonet.snn.lif import LIFNeuron


# =========================================================
# 🔧 Args
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True)
parser.add_argument("--weights",  required=True)
parser.add_argument("--device",   default="cuda")
parser.add_argument("--T",        type=int, default=30,
                    help="Number of SNN timesteps")
parser.add_argument("--rnl_batches", type=int, default=20,
                    help="Calibration batches for RNL")
parser.add_argument("--max_act_path", default=None,
                    help="Path to pre-saved max_activations.pth (skip RNL if provided)")
parser.add_argument("--lif_threshold", type=float, default=0.5,
                    help="LIF neuron firing threshold (lower = more spikes survive deep layers)")
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}")


# =========================================================
# 🔹 STEP 1: Load CNN + dataloader
# =========================================================
print("\n" + "="*55)
print("STEP 1 — Loading CNN")
print("="*55)

cnn_model, val_loader, device = video_run_internal(
    data_dir=args.data_dir,
    weights=args.weights,
    device=device,
    return_objects=True,
)
print("✅ CNN loaded")

# Unwrap DataParallel
if hasattr(cnn_model, "module"):
    print("⚡ Unwrapping DataParallel model")
    cnn_model = cnn_model.module

cnn_model.eval()
cnn_model.to(device)


# =========================================================
# 🔹 STEP 2: CNN Baseline Evaluation (ORIGINAL unmodified model)
# =========================================================
# ⚠️  IMPORTANT for academic integrity:
# CNN baseline is evaluated HERE on the original model, BEFORE any
# BN folding or weight modification. These are the numbers to report
# in the paper as the CNN baseline.
# =========================================================
print("\n" + "="*55)
print("STEP 2 — CNN Baseline Evaluation (original model)")
print("="*55)

cnn_preds, gt_labels = [], []

with torch.no_grad():
    for X, y in tqdm.tqdm(val_loader, desc="CNN inference"):
        X = X.to(device)
        out = cnn_model(X)
        cnn_preds.append(out.squeeze(-1).cpu().numpy())
        gt_labels.append(y.numpy())

cnn_preds  = np.concatenate(cnn_preds)
gt_labels  = np.concatenate(gt_labels)

cnn_mae  = np.mean(np.abs(cnn_preds - gt_labels))
cnn_rmse = math.sqrt(np.mean((cnn_preds - gt_labels) ** 2))
cnn_r2   = sklearn.metrics.r2_score(gt_labels, cnn_preds)

print(f"  CNN  MAE : {cnn_mae:.4f}")
print(f"  CNN  RMSE: {cnn_rmse:.4f}")
print(f"  CNN  R²  : {cnn_r2:.4f}")


# =========================================================
# 🔹 STEP 2b: Deep-copy CNN for SNN conversion
# =========================================================
# The original cnn_model is never touched again after this point.
# All SNN modifications happen only on snn_base.
print("\n⚡ Creating deep copy of CNN for SNN conversion...")
snn_base = copy.deepcopy(cnn_model)
print("⚡ Folding BatchNorm into Conv weights (SNN copy only)...")
fold_batchnorm(snn_base)
n_bn = sum(1 for m in snn_base.modules()
           if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)))
print(f"✅ BN folding done — {n_bn} BN layers remaining (should be 0)")

# =========================================================
# 🔹 STEP 3: Rate Normalization (RNL)
# =========================================================
print("\n" + "="*55)
print("STEP 3 — Rate Normalization (RNL)")
print("="*55)

if args.max_act_path is not None:
    import os
    if os.path.exists(args.max_act_path):
        print(f"⚡ Loading pre-saved activations from {args.max_act_path}")
        print("  ⚠️  Note: saved activations must have been computed AFTER BN folding.")
        print("     If results look wrong, delete this file to re-run RNL.")
        max_activations = torch.load(args.max_act_path, weights_only=False)
        print(f"✅ Loaded {len(max_activations)} layer activations")
    else:
        raise FileNotFoundError(f"max_act_path not found: {args.max_act_path}")
else:
    print(f"⚡ Running RNL over {args.rnl_batches} calibration batches...")
    rnl = RateNormalizer(snn_base, device=device)
    max_activations = rnl.run(val_loader, num_batches=args.rnl_batches)
    save_max_activations(max_activations, "max_activations.pth")
    print(f"✅ RNL done — {len(max_activations)} layers calibrated")
    print("   Top-5 activations:", list(max_activations.items())[:5])


# =========================================================
# 🔹 STEP 4: Convert CNN → SNN
# =========================================================
print("\n" + "="*55)
print("STEP 4 — CNN → SNN Conversion")
print("="*55)

snn_base   = convert_to_snn(snn_base, max_activations, lif_threshold=args.lif_threshold)
snn_model  = SpikingModel(snn_base, time_steps=args.T)
snn_model.to(device)
snn_model.eval()
print(f"✅ SNN ready (T={args.T} timesteps per call)")

# Sanity check: count LIF neurons
n_lif = sum(1 for m in snn_model.modules() if isinstance(m, LIFNeuron))
print(f"   LIF neurons in model: {n_lif}")


# =========================================================
# 🔹 Helper: reset all LIF membrane states
# =========================================================
def reset_snn(model):
    for m in model.modules():
        if hasattr(m, "reset"):
            m.reset()


# =========================================================
# 🔹 STEP 5: SNN Evaluation
# =========================================================
print("\n" + "="*55)
print("STEP 5 — SNN Evaluation")
print("="*55)

snn_preds = []

with torch.no_grad():
    for X, y in tqdm.tqdm(val_loader, desc=f"SNN inference (T={args.T})"):
        X = X.to(device)

        # ── Per-sample normalization to [0,1] ──
        # Each video is normalized independently so spatial patterns are preserved
        B = X.shape[0]
        X_flat = X.view(B, -1)
        X_min = X_flat.min(dim=1, keepdim=True)[0]
        X_max = X_flat.max(dim=1, keepdim=True)[0]
        X_flat = (X_flat - X_min) / (X_max - X_min + 1e-6)
        X = X_flat.view_as(X)
        X = torch.clamp(X, 0.0, 1.0)

        # ── Reset LIF states before every new sample sequence ──
        reset_snn(snn_model)

        # ── SpikingModel samples fresh Bernoulli spikes each timestep ──
        output = snn_model(X)                     # (B, 1), averaged over T

        snn_preds.append(output.squeeze(-1).cpu().numpy())

snn_preds = np.concatenate(snn_preds)

snn_mae  = np.mean(np.abs(snn_preds - gt_labels))
snn_rmse = math.sqrt(np.mean((snn_preds - gt_labels) ** 2))
snn_r2   = sklearn.metrics.r2_score(gt_labels, snn_preds)

# ── Collect per-layer spike rates ──
lif_rates = []
lif_names = []
for name, m in snn_model.named_modules():
    if isinstance(m, LIFNeuron):
        rate = m.get_spike_rate()
        lif_rates.append(rate)
        lif_names.append(name)

avg_spike_rate = np.mean(lif_rates) if lif_rates else float("nan")


# =========================================================
# 🔹 STEP 6: Print comparison table
# =========================================================
print("\n" + "="*55)
print("          RESULTS SUMMARY")
print("="*55)
print(f"  {'Metric':<12} {'CNN':>10} {'SNN':>10}  {'Δ':>10}")
print(f"  {'-'*46}")
print(f"  {'MAE':<12} {cnn_mae:>10.4f} {snn_mae:>10.4f}  {snn_mae - cnn_mae:>+10.4f}")
print(f"  {'RMSE':<12} {cnn_rmse:>10.4f} {snn_rmse:>10.4f}  {snn_rmse - cnn_rmse:>+10.4f}")
print(f"  {'R²':<12} {cnn_r2:>10.4f} {snn_r2:>10.4f}  {snn_r2 - cnn_r2:>+10.4f}")
print(f"  {'-'*46}")
print(f"  Avg spike rate (LIF layers): {avg_spike_rate:.4f}")
print(f"  SNN timesteps (T): {args.T}")
print(f"  Val samples: {len(gt_labels)}")
print("="*55)

# ── Per-layer spike rate breakdown ──
print("\n  Per-layer spike rates:")
for name, rate in zip(lif_names, lif_rates):
    bar = "█" * int(rate * 40)
    status = "🟢" if 0.05 < rate < 0.6 else "🔴"
    print(f"    {status} {name:<40s} {rate:.4f}  {bar}")
print(f"\n  Dead layers (rate < 0.01): "
      f"{sum(1 for r in lif_rates if r < 0.01)}/{len(lif_rates)}")
print(f"  Saturated layers (rate > 0.8): "
      f"{sum(1 for r in lif_rates if r > 0.8)}/{len(lif_rates)}")

# ── Warn if SNN is clearly broken ──
if snn_r2 < 0:
    print("\n⚠️  WARNING: SNN R² < 0 — predictions worse than mean baseline.")
    print("   Try increasing --T (e.g. --T 64) or check LIF threshold.")
elif snn_mae > cnn_mae * 2:
    print("\n⚠️  WARNING: SNN MAE is >2× CNN MAE. Consider increasing --T.")
else:
    print("\n✅ SNN looks reasonable. Consider increasing T for better accuracy.")


# =========================================================
# 🔹 Save results
# =========================================================
torch.save({
    "cnn_preds":  cnn_preds,
    "snn_preds":  snn_preds,
    "gt_labels":  gt_labels,
    "cnn_mae":    cnn_mae,
    "cnn_rmse":   cnn_rmse,
    "cnn_r2":     cnn_r2,
    "snn_mae":    snn_mae,
    "snn_rmse":   snn_rmse,
    "snn_r2":     snn_r2,
    "T":          args.T,
}, "snn_eval_results.pth")

print("\n💾 Results saved to snn_eval_results.pth")
print("✅ Evaluation complete!")
