#!/usr/bin/env python3
"""
Measure T=10 SNN operation counts and calculate analytical energy estimates.

Run from ~/Capstone with the project's .venv activated. This script measures
input activity on the first N test videos (default 30) for each SNN, then calls
the project's own summarize_ops() function to calculate FP32/INT32 estimates.
It does NOT measure physical GPU energy consumption.
"""
import argparse
import gc
import json
from pathlib import Path

import torch

from echonet.utils.snn_common import (
    ENERGY_SETS,
    build_snn,
    get_loader,
    get_norm,
    load_cnn,
    measure_ops,
    summarize_ops,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/home/mluser/datasets/EchoNet-Dynamic")
    p.add_argument("--weights", default="/home/mluser/checkpoints/best.pt")
    p.add_argument("--finetuned_checkpoint", default="ckpt_l4/l4_40ep_best_e24.pt")
    p.add_argument("--T", type=int, default=10)
    p.add_argument("--split", default="test")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--n_videos", type=int, default=30,
                   help="Number of test videos used to estimate activity density")
    p.add_argument("--lif_threshold", type=float, default=0.5)
    p.add_argument("--lif_decay", type=float, default=0.9)
    p.add_argument("--mean", type=float, nargs=3,
                   default=[32.17089, 32.344757, 32.69524])
    p.add_argument("--std", type=float, nargs=3,
                   default=[49.570095, 49.708492, 50.082714])
    p.add_argument("--neurons_per_step", type=float, default=123704812.0,
                   help="Neuron count reported by the completed timestep evaluation")
    p.add_argument("--out", default="energy_T10_measured_ops.json")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Activate the GPU workstation environment.")
    device = torch.device("cuda")
    ckpt = Path(args.finetuned_checkpoint)
    if not ckpt.is_absolute():
        ckpt = Path.cwd() / ckpt
    if not ckpt.exists():
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {ckpt}")

    # Use the same preprocessing values as the completed test evaluation.
    mean, std = get_norm(args)
    loader = get_loader(args, mean, std, device, batch_size=1, limit=args.n_videos)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"T={args.T}; videos used for activity-density measurement: {args.n_videos}")
    print("Energy values are analytical estimates based on ENERGY_SETS:", ENERGY_SETS)

    cnn = load_cnn(args.weights, device)
    results = {
        "settings": {
            "T": args.T,
            "n_videos_for_activity_measurement": args.n_videos,
            "data_dir": args.data_dir,
            "weights": args.weights,
            "finetuned_checkpoint": str(ckpt),
            "mean": args.mean,
            "std": args.std,
            "lif_threshold": args.lif_threshold,
            "lif_decay": args.lif_decay,
            "neurons_per_step": args.neurons_per_step,
            "energy_assumptions_pJ": ENERGY_SETS,
            "note": "Analytical energy estimate, not measured physical energy.",
        },
        "models": {},
    }

    for label, checkpoint in [
        ("Direct SNN", None),
        ("Fine-tuned SNN", str(ckpt)),
    ]:
        print(f"\nMeasuring operation counts: {label} ...", flush=True)
        snn, meta = build_snn(
            cnn, args.lif_threshold, args.lif_decay, device, ckpt_path=checkpoint
        )
        stats = measure_ops(snn, loader, device, args.T, args.n_videos)
        summary = summarize_ops(stats, args.T, args.neurons_per_step)
        # Preserve per-layer density and binary/non-binary classification as audit evidence.
        layer_details = {}
        for name, s in stats.items():
            density = s["nz"] / max(s["el"], 1.0)
            layer_details[name] = {
                "layer_macs_per_step": float(s["macs"] or 0),
                "nonzero_input_fraction": float(density),
                "binary_input": bool(s["binary"]),
                "input_elements_observed": float(s["el"]),
                "nonzero_inputs_observed": float(s["nz"]),
            }
        results["models"][label] = {
            "checkpoint_metadata": meta,
            "operation_summary": summary,
            "layers": layer_details,
        }
        energies = summary["energy_mJ"]
        print(f"  CNN dense MACs: {summary['cnn_macs'] / 1e9:.4f} G")
        print(f"  SNN AC ops:     {summary['snn_ac_ops'] / 1e9:.4f} G")
        print(f"  SNN MAC ops:    {summary['snn_mac_ops'] / 1e9:.4f} G")
        print(f"  Neuron updates: {summary['snn_neuron_updates'] / 1e9:.4f} G")
        for precision, vals in energies.items():
            reduction = (1.0 - vals["ratio"]) * 100.0
            print(
                f"  {precision.upper()}: CNN={vals['cnn']:.3f} mJ, "
                f"SNN={vals['snn']:.3f} mJ, ratio={vals['ratio']:.4f}x, "
                f"estimated reduction={reduction:.2f}%"
            )

        del snn, stats, summary
        gc.collect()
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved detailed operation counts and energy calculations to: {out.resolve()}")
    print("Important: values are analytical estimates, not measured GPU energy.")
    print("The density estimate is based on the first N test videos specified above.")


if __name__ == "__main__":
    main()

