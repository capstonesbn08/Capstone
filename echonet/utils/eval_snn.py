"""
EchoNet-Dynamic CNN -> SNN evaluation.

The evaluator:
1. Loads the trained R(2+1)D-18 CNN.
2. Evaluates the original CNN.
3. Creates a deep copy for SNN conversion.
4. Folds BatchNorm into convolution layers.
5. Optionally performs LIF threshold calibration (RNL).
6. Converts ReLU neurons to LIF neurons.
7. Evaluates the SNN.
8. Reports MAE, RMSE, R2 and spike statistics.

IMPORTANT:
No additional per-video min/max normalization is performed here.
The dataset tensor is passed directly to the CNN/SNN pipeline.
"""

import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import tqdm
import torchvision

import echonet

from echonet.snn.conversion import (
    fold_batchnorm,
    convert_to_snn,
)

from echonet.snn.spiking_model import SpikingModel


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Evaluate EchoNet CNN -> SNN conversion"
)

parser.add_argument(
    "--data_dir",
    required=True,
    help="Path to EchoNet-Dynamic",
)

parser.add_argument(
    "--weights",
    required=True,
    help="Path to trained checkpoint",
)

parser.add_argument(
    "--device",
    default="cuda",
    help="cuda or cpu",
)

parser.add_argument(
    "--T",
    type=int,
    default=30,
    help="Number of SNN timesteps",
)

parser.add_argument(
    "--rnl_batches",
    type=int,
    default=20,
    help="Number of batches for RNL calibration",
)

parser.add_argument(
    "--max_act_path",
    default=None,
    help="Optional saved calibrated-threshold file",
)

parser.add_argument(
    "--lif_threshold",
    type=float,
    default=0.5,
    help="LIF firing threshold (fallback when RNL disabled or a layer has no calibrated value)",
)

parser.add_argument(
    "--lif_decay",
    type=float,
    default=0.9,
    help="LIF membrane decay",
)

parser.add_argument(
    "--max_samples",
    type=int,
    default=None,
    help="Maximum validation videos to evaluate",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=20,
    help="Validation batch size",
)

parser.add_argument(
    "--use_rnl",
    action="store_true",
    help="Enable per-layer LIF threshold calibration",
)

parser.add_argument(
    "--calibration_percentile",
    type=float,
    default=99.9,
    help="Percentile of ReLU activations used to set each layer's calibrated LIF threshold",
)

parser.add_argument(
    "--save_path",
    default="snn_eval_results.pth",
    help="Output results file",
)

parser.add_argument(
    "--anchor_threshold",
    type=float,
    default=0.5,
    help="Rescale calibrated thresholds so their median equals this value. Set negative to disable anchoring.",
)

parser.add_argument(
    "--balance_residual",
    action="store_true",
    help="Calibrate and apply residual-branch magnitude balancing before ReLU->LIF conversion",
)

parser.add_argument(
    "--residual_batches",
    type=int,
    default=20,
    help="Number of batches used for residual-balance calibration",
)

parser.add_argument(
    "--residual_balance_strength",
    type=float,
    default=1.0,
    help="Blend factor for residual balancing: 0.0 = no change, 1.0 = full computed scale",
)

parser.add_argument(
    "--exclude_prefixes",
    default=None,
    help="Comma-separated dotted module-name prefixes to leave as analog ReLU instead of converting to LIF, e.g. 'layer4.1' or 'layer4.0,layer4.1'",
)

args = parser.parse_args()


# ============================================================
# DEVICE
# ============================================================

if args.device.startswith("cuda") and not torch.cuda.is_available():

    print("WARNING: CUDA requested but unavailable.")
    print("Falling back to CPU.")

    device = torch.device("cpu")

else:

    device = torch.device(args.device)


print("=" * 70)
print("EchoNet CNN -> SNN Evaluation")
print("=" * 70)

print(f"Device        : {device}")
print(f"Dataset       : {args.data_dir}")
print(f"Weights       : {args.weights}")
print(f"T             : {args.T}")
print(f"LIF threshold : {args.lif_threshold}")
print(f"LIF decay     : {args.lif_decay}")
print(f"RNL batches   : {args.rnl_batches}")
print(f"Use RNL       : {args.use_rnl}")
print(f"Max samples   : {args.max_samples}")
print("=" * 70)


# ============================================================
# HELPERS
# ============================================================

def unwrap_model(model):
    """Remove DataParallel wrapper if present."""

    if hasattr(model, "module"):

        print("Unwrapping DataParallel model")

        return model.module

    return model


def reset_snn(model):
    """
    Reset stateful SNN neurons.

    Supports both reset() and reset_state()
    depending on the implementation.
    """

    for module in model.modules():

        if hasattr(module, "reset"):

            try:
                module.reset()
            except TypeError:
                pass

        elif hasattr(module, "reset_state"):

            try:
                module.reset_state()
            except TypeError:
                pass


def reset_snn_stats(model):
    """Reset spike counters if supported."""

    for module in model.modules():

        if hasattr(module, "reset_stats"):

            try:
                module.reset_stats()
            except TypeError:
                pass


def collect_spike_rates(model):
    """Collect spike rates from LIF layers."""

    rates = {}

    for name, module in model.named_modules():

        if hasattr(module, "get_spike_rate"):

            try:

                rate = module.get_spike_rate()

                rates[name] = float(rate)

            except Exception:

                pass

    return rates


def regression_metrics(pred, target):

    pred = np.asarray(
        pred,
        dtype=np.float64,
    )

    target = np.asarray(
        target,
        dtype=np.float64,
    )

    error = pred - target

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    ss_res = np.sum(
        (target - pred) ** 2
    )

    ss_tot = np.sum(
        (target - np.mean(target)) ** 2
    )

    if ss_tot == 0:

        r2 = float("nan")

    else:

        r2 = 1.0 - (
            ss_res / ss_tot
        )

    return mae, rmse, r2


def print_metrics(name, pred, target):

    mae, rmse, r2 = regression_metrics(
        pred,
        target,
    )

    print(
        f"  {name:<5} MAE : {mae:.4f}"
    )

    print(
        f"  {name:<5} RMSE: {rmse:.4f}"
    )

    print(
        f"  {name:<5} R2  : {r2:.4f}"
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


# ============================================================
# STEP 1
# LOAD CHECKPOINT
# ============================================================

print()
print("=" * 70)
print("STEP 1 - Loading CNN")
print("=" * 70)


checkpoint = torch.load(
    args.weights,
    map_location=device,
)


print(
    "Checkpoint type:",
    type(checkpoint)
)


if not isinstance(checkpoint, dict):

    raise RuntimeError(
        "Expected checkpoint to be a dictionary."
    )


print(
    "Checkpoint keys:",
    list(checkpoint.keys())
)


# ============================================================
# CREATE EXACT CNN ARCHITECTURE
# ============================================================

print()
print("Creating torchvision R(2+1)D-18...")


model = torchvision.models.video.r2plus1d_18(
    weights=None
)

# EchoNet EF regression uses one output instead of
# the default 400-class Kinetics classifier.
model.fc = nn.Linear(
    model.fc.in_features,
    1,
)

model = unwrap_model(model)


# ============================================================
# LOAD STATE DICT
# ============================================================

if "state_dict" in checkpoint:

    state_dict = checkpoint["state_dict"]

else:

    state_dict = checkpoint


clean_state_dict = {}


for key, value in state_dict.items():

    if key.startswith("module."):

        key = key[len("module."):]

    clean_state_dict[key] = value


missing, unexpected = model.load_state_dict(
    clean_state_dict,
    strict=False,
)


if missing:

    print(
        f"WARNING: {len(missing)} missing keys"
    )

    for key in missing[:10]:

        print(
            "  ",
            key
        )


if unexpected:

    print(
        f"WARNING: {len(unexpected)} unexpected keys"
    )

    for key in unexpected[:10]:

        print(
            "  ",
            key
        )


model = model.to(device)

model.eval()


print("CNN loaded")


# ============================================================
# DATASET
# ============================================================

print()
print("=" * 70)
print("Preparing EchoNet validation dataset")
print("=" * 70)

print()
print("=" * 70)
print("Computing EchoNet training-set normalization")
print("=" * 70)

mean, std = echonet.utils.get_mean_and_std(
    echonet.datasets.Echo(
        root=args.data_dir,
        split="train",
    )
)

print("Training mean:", mean)
print("Training std :", std)

val_dataset = echonet.datasets.Echo(
    root=args.data_dir,
    split="val",
    target_type="EF",
    mean=mean,
    std=std,
    length=32,
    period=2,
)


print(
    f"Validation videos: {len(val_dataset)}"
)


# ============================================================
# LIMIT DATASET
# ============================================================

if args.max_samples is not None:

    if args.max_samples <= 0:

        raise ValueError(
            "--max_samples must be greater than zero"
        )

    n = min(
        args.max_samples,
        len(val_dataset),
    )

    val_dataset = torch.utils.data.Subset(
        val_dataset,
        range(n),
    )

    print(
        f"Limited evaluation to {n} videos"
    )


# ============================================================
# DATALOADER
# ============================================================

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)


# ============================================================
# STEP 2
# CNN BASELINE
# ============================================================

print()
print("=" * 70)
print("STEP 2 - CNN Baseline Evaluation")
print("=" * 70)


cnn_preds = []

cnn_targets = []


with torch.no_grad():

    for X, y in tqdm.tqdm(
        val_loader,
        desc="CNN inference",
    ):

        X = X.to(
            device,
            non_blocking=True,
        )

        y = y.to(
            device,
            non_blocking=True,
        )

        output = model(X)

        output = output.reshape(-1)

        y = y.reshape(-1)

        cnn_preds.extend(
            output.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        cnn_targets.extend(
            y.detach()
            .cpu()
            .numpy()
            .tolist()
        )


cnn_metrics = print_metrics(
    "CNN",
    cnn_preds,
    cnn_targets,
)


# ============================================================
# STEP 3
# CREATE SNN COPY
# ============================================================

print()
print("=" * 70)
print("STEP 3 - Preparing SNN copy")
print("=" * 70)


snn_cnn = copy.deepcopy(
    model
)


snn_cnn = snn_cnn.to(
    device
)


snn_cnn.eval()


print(
    "Creating deep copy of CNN for SNN conversion..."
)


# ============================================================
# STEP 4
# BATCHNORM FOLDING
# ============================================================

print()
print("=" * 70)
print("STEP 4 - BatchNorm Folding")
print("=" * 70)


print(
    "Folding BatchNorm into convolution layers..."
)


snn_cnn = fold_batchnorm(
    snn_cnn
)

# ============================================================
# VERIFY BN FOLDING
# ============================================================

with torch.no_grad():
    test_X, _ = val_dataset[0]

    test_X = torch.from_numpy(test_X).unsqueeze(0).to(device)

    original_out = model(test_X)

    folded_out = snn_cnn(test_X)

    diff = torch.abs(original_out - folded_out)

    print()
    print("=" * 70)
    print("BN FOLDING VERIFICATION")
    print("=" * 70)
    print("Original CNN output :", original_out.item())
    print("Folded CNN output   :", folded_out.item())
    print("Absolute difference :", diff.item())
snn_cnn.eval()


remaining_bn = sum(
    1
    for module in snn_cnn.modules()
    if isinstance(
        module,
        (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
        ),
    )
)


print(
    f"Remaining BatchNorm layers: {remaining_bn}"
)

# ============================================================
# STEP 4b
# RESIDUAL BRANCH BALANCING
# ============================================================

if args.balance_residual:

    print()
    print("=" * 70)
    print("STEP 4b - Residual Branch Balancing")
    print("=" * 70)

    from echonet.snn.residual import (
        calibrate_residual_balance,
        apply_residual_balance,
    )

    residual_calib_batches = []

    with torch.no_grad():

        for batch_idx, (X, _) in enumerate(val_loader):

            if batch_idx >= args.residual_batches:
                break

            residual_calib_batches.append(
                X.to(device, non_blocking=True)
            )

    print(
        f"Collected {len(residual_calib_batches)} "
        "calibration batches"
    )

    raw_scales = calibrate_residual_balance(
        snn_cnn,
        residual_calib_batches,
    )

    if args.residual_balance_strength != 1.0:
        blended_scales = {
            name: 1.0 + args.residual_balance_strength * (s - 1.0)
            for name, s in raw_scales.items()
        }
    else:
        blended_scales = raw_scales

    snn_cnn = apply_residual_balance(
        snn_cnn,
        blended_scales,
    )

else:

    print()
    print("=" * 70)
    print("STEP 4b - Residual Balancing Disabled")
    print("=" * 70)

    print("No residual-branch rebalancing applied.")
# ============================================================
# STEP 5
# RNL (per-layer LIF threshold calibration)
# ============================================================

layer_thresholds = None


if args.use_rnl:

    print()
    print("=" * 70)
    print("STEP 5 - Rate Normalization")
    print("=" * 70)


    if args.max_act_path is not None:

        print(
            "Loading calibrated thresholds:"
        )

        print(
            args.max_act_path
        )


        layer_thresholds = torch.load(
            args.max_act_path,
            map_location="cpu",
        )


        print(
            f"Loaded {len(layer_thresholds)} "
            "calibrated thresholds"
        )


    else:

        from echonet.snn.rnl import calibrate_thresholds


        print(
            f"Running threshold calibration over "
            f"{args.rnl_batches} batches "
            f"(percentile={args.calibration_percentile})..."
        )


        calibration_batches = []


        with torch.no_grad():

            for batch_idx, (X, _) in enumerate(
                val_loader
            ):

                if batch_idx >= args.rnl_batches:

                    break


                calibration_batches.append(
                    X.to(
                        device,
                        non_blocking=True,
                    )
                )


        print(
            f"Collected {len(calibration_batches)} "
            "calibration batches"
        )


        layer_thresholds = calibrate_thresholds(
            snn_cnn,
            calibration_batches,
            percentile=args.calibration_percentile,
            anchor_threshold=(
                args.anchor_threshold
                if args.anchor_threshold >= 0
                else None
            ),
        )


        print(
            f"RNL collected "
            f"{len(layer_thresholds)} layer thresholds"
        )


else:

    print()
    print("=" * 70)
    print("STEP 5 - RNL Disabled")
    print("=" * 70)

    print(
        "No per-layer threshold calibration. "
        f"Using uniform threshold={args.lif_threshold} for all layers."
    )


# ============================================================
# STEP 6
# CNN -> SNN
# ============================================================

print()
print("=" * 70)
print("STEP 6 - CNN -> SNN Conversion")
print("=" * 70)


exclude_prefixes = (
    [p.strip() for p in args.exclude_prefixes.split(",") if p.strip()]
    if args.exclude_prefixes
    else None
)

snn_cnn = convert_to_snn(
    snn_cnn,
    layer_thresholds=layer_thresholds,
    lif_threshold=args.lif_threshold,
    lif_decay=args.lif_decay,
    exclude_prefixes=exclude_prefixes,
)

snn_cnn.eval()


lif_count = sum(
    1
    for module in snn_cnn.modules()
    if hasattr(
        module,
        "get_spike_rate"
    )
)


print(
    f"LIF neurons in model: {lif_count}"
)


# ============================================================
# STEP 7
# SPIKING MODEL
# ============================================================

print()
print("=" * 70)
print("STEP 7 - Creating SpikingModel")
print("=" * 70)


snn_model = SpikingModel(
    snn_cnn,
    time_steps=args.T,
    input_min=-1.0,
    input_max=5.0,
)


snn_model = snn_model.to(
    device
)


snn_model.eval()


print(
    f"SNN ready with T={args.T}"
)


# ============================================================
# INPUT SANITY CHECK
# ============================================================

print()
print("=" * 70)
print("Input Sanity Check")
print("=" * 70)


sample_X, sample_y = val_dataset[0]


print(
    f"Input shape : {tuple(sample_X.shape)}"
)

print(
    f"Input dtype : {sample_X.dtype}"
)

print(
    f"Input min   : {sample_X.min().item():.6f}"
)

print(
    f"Input max   : {sample_X.max().item():.6f}"
)

print(
    f"Input mean  : {sample_X.mean().item():.6f}"
)

print(
    f"Ground truth: {float(sample_y):.6f}"
)


# ============================================================
# STEP 8
# SNN EVALUATION
# ============================================================

print()
print("=" * 70)
print("STEP 8 - SNN Evaluation")
print("=" * 70)


snn_preds = []

snn_targets = []


reset_snn(
    snn_model
)

reset_snn_stats(
    snn_model
)


with torch.no_grad():

    for X, y in tqdm.tqdm(
        val_loader,
        desc=f"SNN inference (T={args.T})",
    ):

        X = X.to(
            device,
            non_blocking=True,
        )

        y = y.to(
            device,
            non_blocking=True,
        )


        # Reset state for every batch.
        reset_snn(
            snn_model
        )


        output = snn_model(
            X
        )


        output = output.reshape(-1)

        y = y.reshape(-1)


        snn_preds.extend(
            output.detach()
            .cpu()
            .numpy()
            .tolist()
        )


        snn_targets.extend(
            y.detach()
            .cpu()
            .numpy()
            .tolist()
        )


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)


snn_metrics = print_metrics(
    "SNN",
    snn_preds,
    snn_targets,
)


print()

print(
    f"{'Metric':<12}"
    f"{'CNN':>12}"
    f"{'SNN':>12}"
    f"{'Delta':>12}"
)

print("-" * 48)


for metric in [
    "mae",
    "rmse",
    "r2",
]:

    cnn_value = cnn_metrics[metric]

    snn_value = snn_metrics[metric]

    delta = (
        snn_value
        - cnn_value
    )

    print(
        f"{metric.upper():<12}"
        f"{cnn_value:>12.4f}"
        f"{snn_value:>12.4f}"
        f"{delta:>12.4f}"
    )


# ============================================================
# SPIKE STATISTICS
# ============================================================

print()
print("=" * 70)
print("SPIKE STATISTICS")
print("=" * 70)


spike_rates = collect_spike_rates(
    snn_model
)


if spike_rates:

    values = list(
        spike_rates.values()
    )


    avg_spike_rate = float(
        np.mean(values)
    )


    dead_layers = [
        (name, rate)
        for name, rate in spike_rates.items()
        if rate < 0.01
    ]


    saturated_layers = [
        (name, rate)
        for name, rate in spike_rates.items()
        if rate > 0.80
    ]


    print(
        f"Average spike rate: "
        f"{avg_spike_rate:.6f}"
    )


    print(
        f"LIF layers: "
        f"{len(values)}"
    )


    print(
        f"Dead layers (<1%): "
        f"{len(dead_layers)}/{len(values)}"
    )


    print(
        f"Saturated layers (>80%): "
        f"{len(saturated_layers)}/{len(values)}"
    )


    print()
    print("Per-layer spike rates:")


    for name, rate in spike_rates.items():

        marker = (
            "🟢"
            if rate >= 0.05
            else "🔴"
        )


        bars = int(
            min(
                rate * 20,
                20
            )
        )


        print(
            f"  {marker} "
            f"{name:<45} "
            f"{rate:.4f} "
            f"{'█' * bars}"
        )


else:

    avg_spike_rate = None

    dead_layers = []

    saturated_layers = []


    print(
        "No spike statistics collected."
    )


# ============================================================
# PREDICTION SANITY CHECK
# ============================================================

print()
print("=" * 70)
print("PREDICTION SANITY CHECK")
print("=" * 70)


cnn_preds_np = np.asarray(
    cnn_preds
)

snn_preds_np = np.asarray(
    snn_preds
)

targets_np = np.asarray(
    cnn_targets
)


print("Ground truth:")

print(
    f"  min  = {targets_np.min():.4f}"
)

print(
    f"  mean = {targets_np.mean():.4f}"
)

print(
    f"  max  = {targets_np.max():.4f}"
)


print()
print("CNN prediction:")

print(
    f"  min  = {cnn_preds_np.min():.4f}"
)

print(
    f"  mean = {cnn_preds_np.mean():.4f}"
)

print(
    f"  max  = {cnn_preds_np.max():.4f}"
)


print()
print("SNN prediction:")

print(
    f"  min  = {snn_preds_np.min():.4f}"
)

print(
    f"  mean = {snn_preds_np.mean():.4f}"
)

print(
    f"  max  = {snn_preds_np.max():.4f}"
)


# ============================================================
# FIRST PREDICTIONS
# ============================================================

print()
print("First predictions:")


for i in range(
    min(
        10,
        len(snn_preds)
    )
):

    print(
        f"  {i:03d} "
        f"GT={targets_np[i]:8.3f} "
        f"CNN={cnn_preds_np[i]:8.3f} "
        f"SNN={snn_preds_np[i]:8.3f}"
    )


# ============================================================
# FINAL WARNING
# ============================================================

if snn_metrics["r2"] < 0:

    print()
    print(
        "WARNING: SNN R2 is negative."
    )

    print(
        "The current SNN is worse than "
        "a mean-value baseline."
    )

    print(
        "Do not assume that increasing T "
        "will solve the problem."
    )

    print(
        "Inspect input encoding, LIF activity "
        "and output scaling first."
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "cnn": {

        "predictions":
            np.asarray(cnn_preds),

        "targets":
            np.asarray(cnn_targets),

        "mae":
            cnn_metrics["mae"],

        "rmse":
            cnn_metrics["rmse"],

        "r2":
            cnn_metrics["r2"],
    },

    "snn": {

        "predictions":
            np.asarray(snn_preds),

        "targets":
            np.asarray(snn_targets),

        "mae":
            snn_metrics["mae"],

        "rmse":
            snn_metrics["rmse"],

        "r2":
            snn_metrics["r2"],
    },

    "settings": {

        "T":
            args.T,

        "lif_threshold":
            args.lif_threshold,

        "lif_decay":
            args.lif_decay,

        "rnl_batches":
            args.rnl_batches,

        "use_rnl":
            args.use_rnl,

        "calibration_percentile":
            args.calibration_percentile,

        "max_samples":
            args.max_samples,

        "batch_size":
            args.batch_size,
    },

    "spike_rates":
        spike_rates,

    "avg_spike_rate":
        avg_spike_rate,

    "dead_layers":
        dead_layers,

    "saturated_layers":
        saturated_layers,

    "layer_thresholds":
        layer_thresholds,
}


torch.save(
    results,
    args.save_path,
)


print()
print(
    "Results saved to:"
)

print(
    f"  {os.path.abspath(args.save_path)}"
)


print()
print("=" * 70)
print("Evaluation complete")
print("=" * 70)
