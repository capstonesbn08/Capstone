"""Timestep ablation: MAE/RMSE/R2, spike rate, latency for several T."""

import argparse

import torch

from echonet.snn.spiking_model import SpikingModel
from echonet.utils.snn_common import (
    add_common_args, get_device, set_seed, load_cnn, build_snn, get_norm,
    get_loader, evaluate_snn, check_same_inputs, lif_no_stats, measure_latency,
    md_table, fmt, save_json,
)


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--finetuned_checkpoint", required=True)
    parser.add_argument("--T_list", type=int, nargs="+", default=[10, 20, 30, 50])
    parser.add_argument("--latency_videos", type=int, default=30)
    parser.add_argument("--also_direct", action="store_true",
                        help="Also run the ablation for the un-fine-tuned (direct) SNN")
    args = parser.parse_args()
    args.out_prefix = args.out_prefix if args.out_prefix != "results" else "test_timestep"

    device = get_device(args)
    set_seed()
    mean, std = get_norm(args)
    loader = get_loader(args, mean, std, device)
    loader1 = get_loader(args, mean, std, device, batch_size=1,
                         limit=args.latency_videos + 3)
    print(f"{args.split} videos: {len(loader.dataset)}")

    cnn = load_cnn(args.weights, device)
    models = []
    if args.also_direct:
        direct, _ = build_snn(cnn, args.lif_threshold, args.lif_decay, device)
        models.append(("Direct SNN", direct))
    ft, meta = build_snn(cnn, args.lif_threshold, args.lif_decay, device,
                         ckpt_path=args.finetuned_checkpoint)
    models.append(("Fine-tuned SNN", ft))
    print(f"Fine-tuned checkpoint meta: {meta}")

    out, checksums = {}, []
    for name, snn in models:
        print(f"\nEvaluating {name} at T = {args.T_list} (single pass) ...")
        res = evaluate_snn(snn, loader, device, args.T_list)
        checksums.append(res["checksum"])

        lat = {}
        for T in sorted(args.T_list):
            spk = SpikingModel(snn, time_steps=T, input_min=-1.0, input_max=5.0).to(device).eval()
            with lif_no_stats():
                lat[T] = measure_latency(lambda X: spk(X), loader1, device, args.latency_videos)

        rows = []
        for T in sorted(args.T_list):
            d = res["T"][T]
            m = d["metrics"]
            rows.append([T, fmt(m["mae"], 3), fmt(m["rmse"], 3), fmt(m["r2"], 4),
                         fmt(d["mean_layer_rate"], 4), fmt(d["overall_rate"], 4),
                         fmt(d["spikes_per_video"] / 1e6, 2),
                         f"{lat[T]['mean_ms']:.1f} ± {lat[T]['std_ms']:.1f}"])
        print(f"\n{name} ({args.split} split)\n")
        print(md_table(["T", "MAE", "RMSE", "R²", "Mean layer rate", "Overall rate",
                        "Spikes/video (M)", "Latency (ms)"], rows))
        out[name] = {str(T): {**{k: v for k, v in res["T"][T].items() if k != "preds"},
                              "latency": lat[T]} for T in args.T_list}

    check_same_inputs(checksums) if len(checksums) > 1 else None
    save_json({"settings": vars(args), "finetuned_meta": meta, "results": out},
              f"{args.out_prefix}_summary.json")


if __name__ == "__main__":
    main()
