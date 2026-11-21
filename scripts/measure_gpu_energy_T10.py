#!/usr/bin/env python3
"""
Measure GPU-board power/energy for CNN, Direct SNN, and Fine-tuned SNN
over the FULL EchoNet-Dynamic test split.

Important:
- Uses nvidia-smi power.draw sampling; results are GPU-board estimates, not
  whole-system energy and not direct per-model hardware energy counters.
- Each model processes every test video exactly once in the timed pass.
- Warm-up is performed separately and excluded from the measured interval.
- Keep the GPU otherwise idle. Concurrent workloads invalidate comparisons.
"""
import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import torch

from echonet.utils.snn_common import (
    build_snn,
    get_loader,
    get_norm,
    load_cnn,
    reset_lifs,
)


class PowerSampler:
    def __init__(self, interval_s=0.1):
        self.interval_s = interval_s
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = None
        self.error = None

    def _sample(self):
        while not self.stop_event.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=3,
                ).strip().splitlines()[0]
                if output and output.upper() != "N/A":
                    self.samples.append((time.monotonic(), float(output)))
            except Exception as exc:
                self.error = str(exc)
            self.stop_event.wait(self.interval_s)

    def start(self):
        self.samples = []
        self.error = None
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
        return list(self.samples)


def summarize_power(samples, t0=None, t1=None):
    selected = [
        (timestamp, watts)
        for timestamp, watts in samples
        if (t0 is None or timestamp >= t0)
        and (t1 is None or timestamp <= t1)
    ]
    if len(selected) < 2:
        return {
            "sample_count": len(selected),
            "avg_power_w": None,
            "gross_energy_j": None,
            "duration_s": 0.0,
        }

    energy_j = 0.0
    for (ta, pa), (tb, pb) in zip(selected, selected[1:]):
        energy_j += (tb - ta) * (pa + pb) / 2.0

    duration_s = selected[-1][0] - selected[0][0]
    avg_power_w = energy_j / duration_s if duration_s > 0 else None
    return {
        "sample_count": len(selected),
        "avg_power_w": avg_power_w,
        "gross_energy_j": energy_j,
        "duration_s": duration_s,
    }


def measure_idle(seconds, sampler):
    sampler.start()
    time.sleep(seconds)
    samples = sampler.stop()
    result = summarize_power(samples)
    if result["sample_count"] < 2 or result["avg_power_w"] is None:
        raise RuntimeError(
            "Insufficient idle power samples. Check nvidia-smi power.draw support."
        )
    return result


def run_inference(model, x, kind, T):
    if kind == "cnn":
        return model(x)

    # Reset the LIF state before each video so videos are independent.
    reset_lifs(model)
    output = None
    for _ in range(T):
        output = model(x)
    return output


def timed_phase(name, model, loader, device, kind, T, warmup, sampler):
    dataset_size = len(loader.dataset)

    # Warm-up on initial test videos, outside the measured interval.
    warmup_count = min(warmup, dataset_size)
    with torch.inference_mode():
        warmup_iter = iter(loader)
        for _ in range(warmup_count):
            try:
                x, _ = next(warmup_iter)
            except StopIteration:
                break
            x = x.to(device, non_blocking=True).float()
            run_inference(model, x, kind, T)
        torch.cuda.synchronize()

        sampler.start()
        # Let the sampler begin collecting before timed inference.
        time.sleep(0.25)
        t0 = time.monotonic()
        inference_count = 0

        # Iterate the entire loader exactly once during measurement.
        for x, _ in loader:
            x = x.to(device, non_blocking=True).float()
            run_inference(model, x, kind, T)
            inference_count += int(x.shape[0])

        torch.cuda.synchronize()
        t1 = time.monotonic()
        samples = sampler.stop()

    result = summarize_power(samples, t0, t1)
    result.update(
        {
            "model": name,
            "inferences": inference_count,
            "expected_test_videos": dataset_size,
            "full_test_split_processed": inference_count == dataset_size,
            "latency_ms_per_inference": (
                (t1 - t0) * 1000.0 / inference_count
                if inference_count
                else None
            ),
            "energy_per_inference_j": (
                result["gross_energy_j"] / inference_count
                if result["gross_energy_j"] is not None and inference_count
                else None
            ),
            "measurement_note": (
                "Gross GPU-board energy during the timed full-test pass; includes "
                "idle/background power and any other GPU activity."
            ),
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Measure sampled GPU-board energy on the full EchoNet test split."
    )
    parser.add_argument(
        "--data_dir",
        default="/home/mluser/datasets/EchoNet-Dynamic",
    )
    parser.add_argument(
        "--weights",
        default="/home/mluser/checkpoints/best.pt",
    )
    parser.add_argument(
        "--finetuned_checkpoint",
        default="ckpt_l4/l4_40ep_best_e24.pt",
    )
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--idle_seconds", type=int, default=30)
    parser.add_argument("--sample_interval", type=float, default=0.1)
    parser.add_argument(
        "--mean",
        type=float,
        nargs=3,
        default=[32.17089, 32.344757, 32.69524],
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs=3,
        default=[49.570095, 49.708492, 50.082714],
    )
    parser.add_argument("--out", default="gpu_energy_T10_full_test.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this script requires a CUDA GPU.")

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)

    checkpoint = Path(args.finetuned_checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = Path.cwd() / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Fine-tuned checkpoint not found: {checkpoint}\n"
            "Set --finetuned_checkpoint to the correct checkpoint path."
        )

    try:
        power_probe = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=3,
        ).strip()
    except Exception as exc:
        raise RuntimeError(
            "Unable to query GPU power using nvidia-smi. Check GPU permissions "
            "and `nvidia-smi -q -d POWER`."
        ) from exc

    print("Power telemetry probe (W):", power_probe)
    print("GPU:", gpu_name)
    print(f"T={args.T}; warm-up videos/model={args.warmup}")
    print(
        "IMPORTANT: keep the GPU otherwise idle. Power is sampled at the GPU-board "
        "level and includes background activity."
    )

    # get_loader expects these attributes on args.
    args.split = "test"
    args.max_samples = None
    args.batch_size = 1

    mean, std = get_norm(args)
    loader = get_loader(
        args, mean, std, device, batch_size=1, limit=None
    )
    test_count = len(loader.dataset)
    print(f"Test videos to evaluate per model: {test_count}")
    if test_count <= 0:
        raise RuntimeError("The test split contains no videos.")

    cnn = load_cnn(args.weights, device)
    sampler = PowerSampler(args.sample_interval)

    print(f"\nMeasuring idle GPU baseline for {args.idle_seconds} seconds ...", flush=True)
    idle = measure_idle(args.idle_seconds, sampler)
    print(json.dumps(idle, indent=2), flush=True)

    results = {
        "settings": vars(args),
        "gpu": gpu_name,
        "idle_baseline": idle,
        "models": {},
        "interpretation": (
            "Power is sampled using nvidia-smi power.draw and integrated over time. "
            "This estimates GPU-board energy, not whole-system energy. The idle-"
            "subtracted value is approximate because sampling is coarse and GPU "
            "power cannot be isolated perfectly by process."
        ),
    }

    # The direct SNN is generated from the CNN weights without a fine-tuned checkpoint.
    model_specs = [
        ("CNN", "cnn", None),
        ("Direct SNN", "snn", None),
        ("Fine-tuned SNN", "snn", str(checkpoint)),
    ]

    for name, kind, snn_checkpoint in model_specs:
        if kind == "cnn":
            model = cnn
        else:
            model, _metadata = build_snn(
                cnn,
                lif_threshold=0.5,
                lif_decay=0.9,
                device=device,
                ckpt_path=snn_checkpoint,
            )

        print(f"\nMeasuring {name} on all {test_count} test videos ...", flush=True)
        result = timed_phase(
            name=name,
            model=model,
            loader=loader,
            device=device,
            kind=kind,
            T=args.T,
            warmup=args.warmup,
            sampler=sampler,
        )

        idle_w = idle.get("avg_power_w")
        average_w = result.get("avg_power_w")
        duration_s = result.get("duration_s")
        if idle_w is not None and average_w is not None and duration_s:
            # Simple baseline subtraction; clamp at zero to avoid negative estimates.
            incremental_w = max(0.0, average_w - idle_w)
            result["idle_baseline_w"] = idle_w
            result["estimated_incremental_power_w"] = incremental_w
            result["estimated_incremental_energy_j"] = incremental_w * duration_s
            result["estimated_incremental_energy_per_inference_j"] = (
                incremental_w * duration_s / result["inferences"]
                if result["inferences"]
                else None
            )

        results["models"][name] = result
        print(json.dumps(result, indent=2), flush=True)

        if kind == "snn":
            del model
            torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved full results to: {out_path.resolve()}")
    print("\nSummary (energy values are estimates):")
    for name, result in results["models"].items():
        print(
            f"{name:16s} | videos={result['inferences']:4d} "
            f"| avg_power={result.get('avg_power_w')} W "
            f"| gross_J/video={result.get('energy_per_inference_j')} "
            f"| incremental_J/video="
            f"{result.get('estimated_incremental_energy_per_inference_j')}"
        )


if __name__ == "__main__":
    main()

