"""
EchoNet-Dynamic CNN baseline evaluation.

Measures:
    Accuracy:
        - MAE
        - RMSE
        - R2

    Efficiency:
        - Inference time per batch
        - Inference time per clip
        - Peak GPU memory
        - Average GPU power
        - GPU energy estimate
        - Spike count = N/A for CNN

Uses:
    - R(2+1)D-18
    - EchoNet-Dynamic TEST split
    - 32 frames
    - period=2
    - training-set normalization
"""

import argparse
import csv
import os
import subprocess
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
import tqdm

import echonet


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Evaluate baseline CNN on EchoNet-Dynamic test set"
)

parser.add_argument(
    "--data_dir",
    default=r"C:\Users\ASUS\Downloads\EchoNet-Dynamic",
)

parser.add_argument(
    "--weights",
    default=r"C:\Users\ASUS\Desktop\cap\Capstone\best.pt",
)

parser.add_argument(
    "--device",
    default="cuda",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=4,
)

parser.add_argument(
    "--num_workers",
    type=int,
    default=0,
)

parser.add_argument(
    "--max_samples",
    type=int,
    default=None,
    help="Optional number of test videos. Default = entire test set.",
)

parser.add_argument(
    "--warmup_batches",
    type=int,
    default=3,
)

args = parser.parse_args()


# ============================================================
# DEVICE
# ============================================================

if args.device.startswith("cuda") and not torch.cuda.is_available():
    print("WARNING: CUDA requested but unavailable. Using CPU.")
    device = torch.device("cpu")
else:
    device = torch.device(args.device)

print("=" * 70)
print("CNN BASELINE EVALUATION")
print("=" * 70)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)

print("Dataset:", args.data_dir)
print("Weights:", args.weights)
print("Batch size:", args.batch_size)
print()


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("=" * 70)
print("STEP 1 - Loading CNN")
print("=" * 70)

checkpoint = torch.load(
    args.weights,
    map_location=device,
    weights_only=False,
)

if not isinstance(checkpoint, dict):
    raise RuntimeError("Expected checkpoint to be a dictionary.")

print("Checkpoint keys:", list(checkpoint.keys()))

if "epoch" in checkpoint:
    print("Checkpoint epoch:", checkpoint["epoch"])

if "r2" in checkpoint:
    print("Recorded checkpoint R2:", checkpoint["r2"])

if "loss" in checkpoint:
    print("Recorded checkpoint loss:", checkpoint["loss"])


# ============================================================
# CREATE R(2+1)D-18
# ============================================================

model = torchvision.models.video.r2plus1d_18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    1,
)


# ============================================================
# LOAD STATE DICT
# ============================================================

state_dict = checkpoint.get(
    "state_dict",
    checkpoint,
)

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
    print("WARNING - missing keys:", len(missing))
    print(missing[:10])

if unexpected:
    print("WARNING - unexpected keys:", len(unexpected))
    print(unexpected[:10])

model = model.to(device)
model.eval()

print("CNN loaded successfully.")
print()


# ============================================================
# DATA NORMALIZATION
# ============================================================

print("=" * 70)
print("STEP 2 - Computing training-set normalization")
print("=" * 70)

train_dataset_for_stats = echonet.datasets.Echo(
    root=args.data_dir,
    split="train",
)

mean, std = echonet.utils.get_mean_and_std(
    train_dataset_for_stats
)

print("Training mean:", mean)
print("Training std :", std)
print()


# ============================================================
# TEST DATASET
# ============================================================

print("=" * 70)
print("STEP 3 - Preparing TEST dataset")
print("=" * 70)

test_dataset = echonet.datasets.Echo(
    root=args.data_dir,
    split="test",
    target_type="EF",
    mean=mean,
    std=std,
    length=32,
    period=2,
)

print("Total test videos:", len(test_dataset))


if args.max_samples is not None:

    n = min(
        args.max_samples,
        len(test_dataset),
    )

    test_dataset = torch.utils.data.Subset(
        test_dataset,
        range(n),
    )

    print("Limited to:", n, "videos")


print()


# ============================================================
# DATALOADER
# ============================================================

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=(device.type == "cuda"),
)


# ============================================================
# GPU POWER MONITOR
# ============================================================

power_samples = []
monitor_stop = threading.Event()


def read_gpu_power():

    try:

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )

        value = float(
            result.stdout.strip().splitlines()[0]
        )

        return value

    except Exception:
        return None


def power_monitor():

    while not monitor_stop.is_set():

        value = read_gpu_power()

        if value is not None:
            power_samples.append(
                (time.perf_counter(), value)
            )

        time.sleep(0.1)


# ============================================================
# WARMUP
# ============================================================

print("=" * 70)
print("STEP 4 - GPU warmup")
print("=" * 70)

if device.type == "cuda":

    with torch.no_grad():

        for batch_index, (X, y) in enumerate(test_loader):

            if batch_index >= args.warmup_batches:
                break

            X = X.to(
                device,
                non_blocking=True,
            )

            _ = model(X)

    torch.cuda.synchronize()

print("Warmup complete.")
print()


# ============================================================
# RESET GPU MEMORY
# ============================================================

if device.type == "cuda":

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


# ============================================================
# EVALUATION
# ============================================================

print("=" * 70)
print("STEP 5 - CNN TEST EVALUATION")
print("=" * 70)

predictions = []
targets = []

batch_times = []
clip_times = []

total_clips = 0


# Start power monitoring

if device.type == "cuda":

    power_samples.clear()

    monitor_stop.clear()

    monitor_thread = threading.Thread(
        target=power_monitor,
        daemon=True,
    )

    monitor_thread.start()

else:

    monitor_thread = None


# Overall inference timer

if device.type == "cuda":
    torch.cuda.synchronize()

overall_start = time.perf_counter()


with torch.no_grad():

    for X, y in tqdm.tqdm(
        test_loader,
        desc="CNN TEST inference",
    ):

        batch_size_actual = X.shape[0]

        X = X.to(
            device,
            non_blocking=True,
        )

        y = y.to(
            device,
            non_blocking=True,
        )


        # ----------------------------------------------------
        # GPU TIMING
        # ----------------------------------------------------

        if device.type == "cuda":

            start_event = torch.cuda.Event(
                enable_timing=True
            )

            end_event = torch.cuda.Event(
                enable_timing=True
            )

            start_event.record()

            output = model(X)

            end_event.record()

            torch.cuda.synchronize()

            elapsed_ms = start_event.elapsed_time(
                end_event
            )

        else:

            start_cpu = time.perf_counter()

            output = model(X)

            elapsed_ms = (
                time.perf_counter() - start_cpu
            ) * 1000


        # ----------------------------------------------------
        # STORE PREDICTIONS
        # ----------------------------------------------------

        output = output.reshape(-1)

        y = y.reshape(-1)

        predictions.extend(
            output.detach().cpu().numpy().tolist()
        )

        targets.extend(
            y.detach().cpu().numpy().tolist()
        )


        # ----------------------------------------------------
        # TIMING
        # ----------------------------------------------------

        batch_times.append(
            elapsed_ms
        )

        clip_times.extend(
            [elapsed_ms / batch_size_actual]
            * batch_size_actual
        )

        total_clips += batch_size_actual


# Overall end

if device.type == "cuda":
    torch.cuda.synchronize()

overall_end = time.perf_counter()


# Stop power monitor

if monitor_thread is not None:

    monitor_stop.set()
    monitor_thread.join(timeout=2)


# ============================================================
# METRICS
# ============================================================

predictions = np.asarray(
    predictions,
    dtype=np.float64,
)

targets = np.asarray(
    targets,
    dtype=np.float64,
)


error = predictions - targets


mae = np.mean(
    np.abs(error)
)

rmse = np.sqrt(
    np.mean(error ** 2)
)

ss_res = np.sum(
    error ** 2
)

ss_tot = np.sum(
    (targets - np.mean(targets)) ** 2
)

if ss_tot == 0:

    r2 = float("nan")

else:

    r2 = 1.0 - (
        ss_res / ss_tot
    )


# ============================================================
# TIMING METRICS
# ============================================================

mean_batch_time_ms = np.mean(
    batch_times
)

median_batch_time_ms = np.median(
    batch_times
)

mean_clip_time_ms = np.mean(
    clip_times
)

median_clip_time_ms = np.median(
    clip_times
)

total_inference_seconds = (
    overall_end - overall_start
)


# ============================================================
# MEMORY
# ============================================================

if device.type == "cuda":

    peak_memory_mb = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 2)
    )

else:

    peak_memory_mb = float("nan")


# ============================================================
# POWER / ENERGY
# ============================================================

if len(power_samples) > 0:

    powers = np.asarray(
        [p for _, p in power_samples],
        dtype=np.float64,
    )

    average_power_w = np.mean(
        powers
    )

    max_power_w = np.max(
        powers
    )

    min_power_w = np.min(
        powers
    )

    # Approximate GPU energy.
    #
    # E = average power × elapsed inference time

    gpu_energy_j = (
        average_power_w
        * total_inference_seconds
    )

else:

    average_power_w = float("nan")
    max_power_w = float("nan")
    min_power_w = float("nan")
    gpu_energy_j = float("nan")


# Energy per clip

if total_clips > 0:

    energy_per_clip_j = (
        gpu_energy_j
        / total_clips
    )

else:

    energy_per_clip_j = float("nan")


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = "cnn_baseline_results.csv"

with open(
    results_path,
    "w",
    newline="",
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "metric",
            "value",
        ]
    )

    writer.writerow(
        ["MAE", mae]
    )

    writer.writerow(
        ["RMSE", rmse]
    )

    writer.writerow(
        ["R2", r2]
    )

    writer.writerow(
        [
            "Inference time per batch (ms)",
            mean_batch_time_ms,
        ]
    )

    writer.writerow(
        [
            "Inference time per clip (ms)",
            mean_clip_time_ms,
        ]
    )

    writer.writerow(
        [
            "Peak GPU memory (MB)",
            peak_memory_mb,
        ]
    )

    writer.writerow(
        [
            "Average GPU power (W)",
            average_power_w,
        ]
    )

    writer.writerow(
        [
            "Maximum GPU power (W)",
            max_power_w,
        ]
    )

    writer.writerow(
        [
            "Minimum GPU power (W)",
            min_power_w,
        ]
    )

    writer.writerow(
        [
            "Total GPU energy (J)",
            gpu_energy_j,
        ]
    )

    writer.writerow(
        [
            "GPU energy per clip (J)",
            energy_per_clip_j,
        ]
    )

    writer.writerow(
        [
            "Spike count",
            "N/A - CNN",
        ]
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("CNN BASELINE RESULTS")
print("=" * 70)

print(f"Test clips                 : {total_clips}")

print()
print("ACCURACY")
print(f"MAE                        : {mae:.4f}")
print(f"RMSE                       : {rmse:.4f}")
print(f"R2                         : {r2:.4f}")

print()
print("EFFICIENCY")
print(
    f"Mean inference / batch     : "
    f"{mean_batch_time_ms:.4f} ms"
)

print(
    f"Median inference / batch   : "
    f"{median_batch_time_ms:.4f} ms"
)

print(
    f"Mean inference / clip      : "
    f"{mean_clip_time_ms:.4f} ms"
)

print(
    f"Median inference / clip    : "
    f"{median_clip_time_ms:.4f} ms"
)

print(
    f"Peak GPU memory            : "
    f"{peak_memory_mb:.2f} MB"
)

print(
    f"Average GPU power          : "
    f"{average_power_w:.4f} W"
)

print(
    f"Maximum GPU power          : "
    f"{max_power_w:.4f} W"
)

print(
    f"Total GPU energy           : "
    f"{gpu_energy_j:.4f} J"
)

print(
    f"GPU energy / clip          : "
    f"{energy_per_clip_j:.6f} J"
)

print(
    "Spike count                : N/A (CNN)"
)

print()
print("Results saved to:")
print(os.path.abspath(results_path))

print("=" * 70)
print("Evaluation complete")
print("=" * 70)