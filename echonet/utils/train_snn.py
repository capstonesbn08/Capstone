"""
Surrogate-gradient fine-tuning for the converted EchoNet SNN.

Starts from a BN-folded, ReLU->LIF converted network (same pipeline
as eval_snn.py) and continues training end-to-end with a surrogate
gradient through the spike nonlinearity, so weights adapt to the
actual spiking-input distributions they receive at inference —
instead of remaining frozen at values learned for continuous ReLU
activations.

Conservative defaults are chosen deliberately:
  - Smaller T during training (T_train) than at inference (T_eval),
    to keep the BPTT unroll memory-tractable.
  - Early layers (stem, layer1, layer2) frozen by default; only
    layer3, layer4, and fc are fine-tuned, cutting compute and
    reducing instability risk.
  - Gradient clipping enabled by default.

Get ONE stable run working with small settings before scaling up
T_train, unfreezing more layers, or training more epochs.
"""

import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import torchvision

import echonet

from echonet.snn.conversion import fold_batchnorm, convert_to_snn
from echonet.snn.spiking_model import SpikingModel
from echonet.snn.lif import LIFNeuron


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", required=True)
parser.add_argument("--weights", required=True, help="Starting CNN checkpoint")
parser.add_argument("--device", default="cuda")

parser.add_argument("--T_train", type=int, default=10, help="Timesteps during training (BPTT unroll length)")
parser.add_argument("--T_eval", type=int, default=50, help="Timesteps used for periodic validation")

parser.add_argument("--lif_threshold", type=float, default=0.5)
parser.add_argument("--lif_decay", type=float, default=0.9)
parser.add_argument("--surrogate_scale", type=float, default=10.0)

parser.add_argument("--freeze_prefixes", default="stem,layer1,layer2",
                     help="Comma-separated module-name prefixes to FREEZE (no gradient updates). Empty string = train everything.")

parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=4,
                     help="Small default: T_train unroll multiplies memory use substantially")
parser.add_argument("--grad_clip", type=float, default=1.0)

parser.add_argument("--train_samples", type=int, default=None,
                     help="Optional cap on training videos, for a fast first test run")
parser.add_argument("--val_samples", type=int, default=100,
                     help="Videos used for periodic validation (subset, for speed)")
parser.add_argument("--val_every", type=int, default=1, help="Validate every N epochs")

parser.add_argument("--save_path", default="snn_finetuned.pt")
parser.add_argument("--log_every", type=int, default=10, help="Print training loss every N batches")

parser.add_argument(
    "--resume_from",
    default=None,
    help="Path to a previously fine-tuned checkpoint (from this script) to resume training from, instead of starting fresh from --weights",
)

args = parser.parse_args()


# ============================================================
# DEVICE
# ============================================================

device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
print(f"Device: {device}")
print(f"T_train={args.T_train}  T_eval={args.T_eval}  lr={args.lr}  batch_size={args.batch_size}")
print(f"Freeze prefixes: {args.freeze_prefixes}")


# ============================================================
# LOAD STARTING CNN
# ============================================================

checkpoint = torch.load(args.weights, map_location=device)
state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

clean_state_dict = {}
for key, value in state_dict.items():
    if key.startswith("module."):
        key = key[len("module."):]
    clean_state_dict[key] = value

base_model = torchvision.models.video.r2plus1d_18(weights=None)
base_model.fc = nn.Linear(base_model.fc.in_features, 1)
base_model.load_state_dict(clean_state_dict, strict=False)
base_model = base_model.to(device)
base_model.eval()

print("Starting CNN loaded")


# ============================================================
# DATASET
# ============================================================

mean, std = echonet.utils.get_mean_and_std(
    echonet.datasets.Echo(root=args.data_dir, split="train")
)
print("Train mean:", mean, " std:", std)

train_dataset = echonet.datasets.Echo(
    root=args.data_dir, split="train", target_type="EF",
    mean=mean, std=std, length=32, period=2,
)

val_dataset = echonet.datasets.Echo(
    root=args.data_dir, split="val", target_type="EF",
    mean=mean, std=std, length=32, period=2,
)

if args.train_samples is not None:
    n = min(args.train_samples, len(train_dataset))
    train_dataset = torch.utils.data.Subset(train_dataset, range(n))

val_subset = torch.utils.data.Subset(val_dataset, range(min(args.val_samples, len(val_dataset))))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=0, pin_memory=(device.type == "cuda"),
)
val_loader = torch.utils.data.DataLoader(
    val_subset, batch_size=args.batch_size, shuffle=False,
    num_workers=0, pin_memory=(device.type == "cuda"),
)

print(f"Train videos: {len(train_dataset)}   Val (subset) videos: {len(val_subset)}")


# ============================================================
# BUILD TRAINABLE SNN
# ============================================================

snn_cnn = copy.deepcopy(base_model)
snn_cnn = snn_cnn.to(device)

snn_cnn = fold_batchnorm(snn_cnn)

snn_cnn = convert_to_snn(
    snn_cnn,
    layer_thresholds=None,
    lif_threshold=args.lif_threshold,
    lif_decay=args.lif_decay,
    surrogate=True,
    surrogate_scale=args.surrogate_scale,
    detach_reset=True,
)

if args.resume_from is not None:
    print(f"Resuming from fine-tuned checkpoint: {args.resume_from}")
    resume_ckpt = torch.load(args.resume_from, map_location=device)
    resume_state = resume_ckpt["state_dict"] if "state_dict" in resume_ckpt else resume_ckpt
    snn_cnn.load_state_dict(resume_state, strict=True)
    prev_r2 = resume_ckpt.get("val_r2", None)
    if prev_r2 is not None:
        print(f"  Previous checkpoint's recorded val R2: {prev_r2:.4f}")

freeze_prefixes = [p.strip() for p in args.freeze_prefixes.split(",") if p.strip()]

n_frozen = 0
n_trainable = 0

for name, param in snn_cnn.named_parameters():
    if any(name == p or name.startswith(p + ".") for p in freeze_prefixes):
        param.requires_grad = False
        n_frozen += 1
    else:
        param.requires_grad = True
        n_trainable += 1

print(f"Parameters: {n_trainable} trainable tensors, {n_frozen} frozen tensors")

snn_model_train = SpikingModel(snn_cnn, time_steps=args.T_train, input_min=-1.0, input_max=5.0).to(device)
snn_model_eval = SpikingModel(snn_cnn, time_steps=args.T_eval, input_min=-1.0, input_max=5.0).to(device)
# Both wrap the SAME underlying snn_cnn — only the timestep count differs.


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


def set_lif_training_mode(model, training):
    for module in model.modules():
        if isinstance(module, LIFNeuron):
            module.train(training)


def regression_metrics(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    error = pred - target
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    r2 = float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return mae, rmse, r2


def validate(model, loader, T):
    model.eval()
    set_lif_training_mode(snn_cnn, False)
    preds, targets = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            reset_snn(model)
            out = model(X).reshape(-1)
            preds.extend(out.cpu().numpy().tolist())
            targets.extend(y.reshape(-1).cpu().numpy().tolist())
    mae, rmse, r2 = regression_metrics(preds, targets)
    return mae, rmse, r2


# ============================================================
# OPTIMIZER
# ============================================================

trainable_params = [p for p in snn_cnn.parameters() if p.requires_grad]
optimizer = optim.Adam(trainable_params, lr=args.lr)
loss_fn = nn.MSELoss()


# ============================================================
# INITIAL VALIDATION (before any fine-tuning — sanity baseline)
# ============================================================

print()
print("=" * 70)
print("Baseline validation BEFORE fine-tuning (T_eval)")
print("=" * 70)

mae0, rmse0, r20 = validate(snn_model_eval, val_loader, args.T_eval)
print(f"  MAE={mae0:.4f}  RMSE={rmse0:.4f}  R2={r20:.4f}")

best_r2 = r20


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(1, args.epochs + 1):

    print()
    print("=" * 70)
    print(f"EPOCH {epoch}/{args.epochs}")
    print("=" * 70)

    snn_model_train.train()
    set_lif_training_mode(snn_cnn, True)

    epoch_start = time.time()
    running_loss = 0.0
    n_batches = 0

    for batch_idx, (X, y) in enumerate(tqdm.tqdm(train_loader, desc=f"Train epoch {epoch}")):

        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).reshape(-1).float()

        reset_snn(snn_model_train)

        optimizer.zero_grad()

        pred = snn_model_train(X).reshape(-1)

        loss = loss_fn(pred, y)
        loss.backward()

        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)

        # NaN/inf guard: skip a corrupted update rather than poisoning
        # the model. If this fires often, lr is too high or T_train
        # too long — reduce one of them.
        bad_grad = any(
            (p.grad is not None) and (not torch.isfinite(p.grad).all())
            for p in trainable_params
        )
        if bad_grad:
            print(f"  ⚠️  Non-finite gradient at batch {batch_idx} — skipping update")
            optimizer.zero_grad()
            continue

        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % args.log_every == 0:
            avg_loss = running_loss / n_batches
            print(f"  batch {batch_idx+1}: loss={loss.item():.4f}  running_avg_loss={avg_loss:.4f}")

    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch} done in {epoch_time/60:.1f} min. Avg training loss: {running_loss/max(n_batches,1):.4f}")

    if epoch % args.val_every == 0:

        print(f"Validating (T_eval={args.T_eval})...")
        mae, rmse, r2 = validate(snn_model_eval, val_loader, args.T_eval)
        print(f"  VAL  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}   (baseline R2 was {r20:.4f})")

        if r2 > best_r2:
            best_r2 = r2
            torch.save(
                {
                    "state_dict": snn_cnn.state_dict(),
                    "epoch": epoch,
                    "val_mae": mae,
                    "val_rmse": rmse,
                    "val_r2": r2,
                    "settings": vars(args),
                },
                args.save_path,
            )
            print(f"  ✅ New best R2={r2:.4f} — saved to {args.save_path}")

    # back to training mode for next epoch
    snn_model_train.train()
    set_lif_training_mode(snn_cnn, True)

print()
print("=" * 70)
print("Training complete")
print(f"Best validation R2 achieved: {best_r2:.4f}  (pre-finetune baseline: {r20:.4f})")
print(f"Best checkpoint saved to: {os.path.abspath(args.save_path)}")
print("=" * 70)