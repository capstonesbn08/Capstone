"""
Diagnostic script: log the SNN's per-timestep prediction trajectory
for a handful of sample videos, instead of only the final T-averaged
output. This tells us whether the SNN accumulates useful signal over
time, or settles/flatlines almost immediately.

Uses the same conversion pipeline as eval_snn.py (BN fold -> uniform
threshold LIF), just with instrumented inference.
"""

import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
import torchvision

import echonet

from echonet.snn.conversion import fold_batchnorm, convert_to_snn


parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", required=True)
parser.add_argument("--weights", required=True)
parser.add_argument("--device", default="cuda")
parser.add_argument("--T", type=int, default=50)
parser.add_argument("--lif_threshold", type=float, default=0.5)
parser.add_argument("--lif_decay", type=float, default=0.9)
parser.add_argument("--num_videos", type=int, default=5)
parser.add_argument("--save_path", default="trajectory_probe.pth")

args = parser.parse_args()


# ============================================================
# DEVICE
# ============================================================

if args.device.startswith("cuda") and not torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = torch.device(args.device)

print(f"Device: {device}")


# ============================================================
# LOAD CNN
# ============================================================

checkpoint = torch.load(args.weights, map_location=device)
state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

clean_state_dict = {}
for key, value in state_dict.items():
    if key.startswith("module."):
        key = key[len("module."):]
    clean_state_dict[key] = value

model = torchvision.models.video.r2plus1d_18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(clean_state_dict, strict=False)
model = model.to(device)
model.eval()

print("CNN loaded")


# ============================================================
# DATASET
# ============================================================

mean, std = echonet.utils.get_mean_and_std(
    echonet.datasets.Echo(root=args.data_dir, split="train")
)

val_dataset = echonet.datasets.Echo(
    root=args.data_dir,
    split="val",
    target_type="EF",
    mean=mean,
    std=std,
    length=32,
    period=2,
)

print(f"Validation videos: {len(val_dataset)}")


# ============================================================
# BUILD SNN (same pipeline as eval_snn.py, uniform threshold)
# ============================================================

snn_cnn = copy.deepcopy(model)
snn_cnn = snn_cnn.to(device)
snn_cnn.eval()

snn_cnn = fold_batchnorm(snn_cnn)

snn_cnn = convert_to_snn(
    snn_cnn,
    layer_thresholds=None,
    lif_threshold=args.lif_threshold,
    lif_decay=args.lif_decay,
)

snn_cnn.eval()

print(
    f"SNN built: threshold={args.lif_threshold}, "
    f"decay={args.lif_decay}, T={args.T}"
)


# ============================================================
# HELPERS
# ============================================================

def reset_snn(model):
    for module in model.modules():
        if hasattr(module, "reset"):
            try:
                module.reset()
            except TypeError:
                pass


def run_trajectory(model, x, T):
    """
    Run the converted SNN for T timesteps on a single video,
    presenting the same input every step (matching SpikingModel's
    behavior), and record the raw prediction at every step
    (not just the running average).
    """

    reset_snn(model)

    per_step_outputs = []

    with torch.no_grad():
        for t in range(T):
            out = model(x)
            per_step_outputs.append(out.reshape(-1).item())

    per_step_outputs = np.array(per_step_outputs)

    # Running average trajectory: this is what eval_snn.py's final
    # output corresponds to at each cutoff T' <= T.
    running_avg = np.cumsum(per_step_outputs) / np.arange(1, T + 1)

    return per_step_outputs, running_avg


# ============================================================
# RUN PROBE ON A FEW VIDEOS
# ============================================================

results = []

for i in range(min(args.num_videos, len(val_dataset))):

    X, y = val_dataset[i]
    X = torch.from_numpy(X).unsqueeze(0).to(device)
    gt = float(y)

    with torch.no_grad():
        cnn_pred = model(X).reshape(-1).item()

    per_step, running_avg = run_trajectory(snn_cnn, X, args.T)

    print()
    print("=" * 70)
    print(f"Video {i}  GT={gt:.3f}  CNN={cnn_pred:.3f}")
    print("=" * 70)

    print("Per-step raw prediction (first 20 steps):")
    print(
        "  " + "  ".join(f"{v:6.2f}" for v in per_step[:20])
    )

    print("Running average (first 20 steps):")
    print(
        "  " + "  ".join(f"{v:6.2f}" for v in running_avg[:20])
    )

    print(
        f"Final running average (t=T={args.T}): {running_avg[-1]:.3f}"
    )

    # Rough settling check: how much does the running average move
    # in the second half of the simulation vs the first half?
    half = args.T // 2
    early_avg = running_avg[:half].mean()
    late_avg = running_avg[half:].mean()

    print(
        f"Early-half avg (t<{half}) = {early_avg:.3f}   "
        f"Late-half avg (t>={half}) = {late_avg:.3f}   "
        f"Drift = {abs(late_avg - early_avg):.3f}"
    )

    results.append(
        {
            "index": i,
            "gt": gt,
            "cnn_pred": cnn_pred,
            "per_step": per_step,
            "running_avg": running_avg,
        }
    )

torch.save(results, args.save_path)
print()
print(f"Saved trajectory data to {args.save_path}")