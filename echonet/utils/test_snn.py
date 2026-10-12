"""
Final evaluation of a fine-tuned SNN checkpoint on the held-out TEST split.
Loads a fine-tuned checkpoint (from train_snn.py) into the converted SNN
architecture and reports MAE/RMSE/R2 at the chosen T_eval.
"""

import argparse
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import tqdm
import torchvision

import echonet

from echonet.snn.conversion import fold_batchnorm, convert_to_snn
from echonet.snn.spiking_model import SpikingModel


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True)
parser.add_argument("--weights", required=True,
                    help="Original CNN checkpoint (used to build the architecture)")
parser.add_argument("--finetuned_checkpoint", required=True,
                    help="Fine-tuned SNN checkpoint from train_snn.py")
parser.add_argument("--device", default="cuda")
parser.add_argument("--T_eval", type=int, default=50)
parser.add_argument("--lif_threshold", type=float, default=0.5)
parser.add_argument("--lif_decay", type=float, default=0.9)
parser.add_argument("--batch_size", type=int, default=20)
parser.add_argument("--mean", type=float, nargs=3, default=None,
                    help="Optional: 3 channel means copied from the training log")
parser.add_argument("--std", type=float, nargs=3, default=None,
                    help="Optional: 3 channel stds copied from the training log")
parser.add_argument("--save_path", default="snn_test_results.pth")
args = parser.parse_args()

device = torch.device(
    args.device if (args.device.startswith("cuda") and torch.cuda.is_available())
    else "cpu"
)
print(f"Device: {device}")


# ---- Original CNN architecture ----
checkpoint = torch.load(args.weights, map_location=device)
state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
clean_state_dict = {
    (k[len("module."):] if k.startswith("module.") else k): v
    for k, v in state_dict.items()
}

base_model = torchvision.models.video.r2plus1d_18(weights=None)
base_model.fc = nn.Linear(base_model.fc.in_features, 1)
base_model.load_state_dict(clean_state_dict, strict=False)
base_model = base_model.to(device)
base_model.eval()
print("Base CNN architecture loaded")


# ---- Build SNN (BN fold + LIF convert) ----
snn_cnn = copy.deepcopy(base_model).to(device)
snn_cnn = fold_batchnorm(snn_cnn)
snn_cnn = convert_to_snn(
    snn_cnn,
    layer_thresholds=None,
    lif_threshold=args.lif_threshold,
    lif_decay=args.lif_decay,
    surrogate=False,
)

print(f"Loading fine-tuned checkpoint: {args.finetuned_checkpoint}")
ft_ckpt = torch.load(args.finetuned_checkpoint, map_location=device)
ft_state = ft_ckpt["state_dict"] if "state_dict" in ft_ckpt else ft_ckpt
snn_cnn.load_state_dict(ft_state, strict=True)
snn_cnn.eval()

reported_r2 = ft_ckpt.get("val_r2", None)
if reported_r2 is not None:
    print(f"  Checkpoint's recorded VAL R2 (for reference): {reported_r2:.4f}")


# ---- Normalization ----
if args.mean is not None and args.std is not None:
    mean = np.array(args.mean, dtype=np.float32)
    std = np.array(args.std, dtype=np.float32)
    print("Using mean/std supplied on the command line")
else:
    mean, std = echonet.utils.get_mean_and_std(
        echonet.datasets.Echo(root=args.data_dir, split="train")
    )
    print("Computed mean/std from a random train subsample")
print("Mean:", mean, " Std:", std)


# ---- TEST split ----
test_dataset = echonet.datasets.Echo(
    root=args.data_dir, split="test", target_type="EF",
    mean=mean, std=std, length=32, period=2,
)
print(f"Test videos: {len(test_dataset)}")

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=0, pin_memory=(device.type == "cuda"),
)

snn_model = SpikingModel(
    snn_cnn, time_steps=args.T_eval, input_min=-1.0, input_max=5.0
).to(device)
snn_model.eval()


def reset_snn(model):
    for module in model.modules():
        if hasattr(module, "reset"):
            try:
                module.reset()
            except TypeError:
                pass


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


# ---- Evaluate ----
preds, targets = [], []

with torch.no_grad():
    for X, y in tqdm.tqdm(test_loader, desc=f"SNN TEST inference (T={args.T_eval})"):
        X = X.to(device, non_blocking=True)
        reset_snn(snn_model)
        out = snn_model(X).reshape(-1)
        preds.extend(out.cpu().numpy().tolist())
        targets.extend(y.reshape(-1).cpu().numpy().tolist())

mae, rmse, r2 = regression_metrics(preds, targets)

print()
print("=" * 70)
print("FINAL TEST SET RESULTS")
print("=" * 70)
print(f"  Test videos: {len(test_dataset)}")
print(f"  MAE  = {mae:.4f}")
print(f"  RMSE = {rmse:.4f}")
print(f"  R2   = {r2:.4f}")
print("=" * 70)

torch.save(
    {
        "predictions": np.asarray(preds),
        "targets": np.asarray(targets),
        "mae": mae, "rmse": rmse, "r2": r2,
        "T_eval": args.T_eval,
        "finetuned_checkpoint": args.finetuned_checkpoint,
        "mean": np.asarray(mean), "std": np.asarray(std),
    },
    args.save_path,
)
print(f"\nResults saved to: {args.save_path}")
