"""CNN vs Direct SNN vs Fine-tuned SNN on the test split: accuracy + efficiency."""

import argparse

import numpy as np
import torch

from echonet.snn.spiking_model import SpikingModel
from echonet.utils.snn_common import (
    add_common_args, get_device, set_seed, load_cnn, build_snn, get_norm,
    get_loader, evaluate_cnn, evaluate_snn, check_same_inputs, lif_no_stats,
    measure_latency, measure_gpu_memory, weights_mib, measure_ops, summarize_ops,
    md_table, fmt, save_json,
)


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--finetuned_checkpoint", required=True)
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--latency_videos", type=int, default=30)
    parser.add_argument("--ops_videos", type=int, default=20)
    args = parser.parse_args()
    args.out_prefix = args.out_prefix if args.out_prefix != "results" else "test_compare"

    device = get_device(args)
    set_seed()
    mean, std = get_norm(args)
    loader = get_loader(args, mean, std, device)
    print(f"{args.split} videos: {len(loader.dataset)}")

    cnn = load_cnn(args.weights, device)
    snn_direct, _ = build_snn(cnn, args.lif_threshold, args.lif_decay, device)
    snn_ft, meta = build_snn(cnn, args.lif_threshold, args.lif_decay, device,
                             ckpt_path=args.finetuned_checkpoint)
    print(f"Fine-tuned checkpoint meta: {meta}")

    # ---------------- accuracy ----------------
    print("\nEvaluating CNN ...")
    r_cnn = evaluate_cnn(cnn, loader, device)
    print("Evaluating direct SNN ...")
    r_dir = evaluate_snn(snn_direct, loader, device, [args.T])
    print("Evaluating fine-tuned SNN ...")
    r_ft = evaluate_snn(snn_ft, loader, device, [args.T])
    paired = check_same_inputs([r_cnn["checksum"], r_dir["checksum"], r_ft["checksum"]])

    T = args.T
    acc_rows = []
    for name, m in [("CNN", r_cnn["metrics"]),
                    (f"Direct SNN (T={T})", r_dir["T"][T]["metrics"]),
                    (f"Fine-tuned SNN (T={T})", r_ft["T"][T]["metrics"])]:
        acc_rows.append([name, fmt(m["mae"], 3), fmt(m["rmse"], 3), fmt(m["r2"], 4)])

    # ---------------- efficiency ----------------
    print("\nMeasuring latency / memory / operation counts ...")
    loader1 = get_loader(args, mean, std, device, batch_size=1,
                         limit=max(args.latency_videos + 3, args.ops_videos))
    X1 = next(iter(loader1))[0].to(device).float()

    cnn_lat = measure_latency(lambda X: cnn(X), loader1, device, args.latency_videos)
    cnn_mem = measure_gpu_memory(lambda X: cnn(X), X1, device)

    def snn_efficiency(snn, res):
        spk = SpikingModel(snn, time_steps=T, input_min=-1.0, input_max=5.0).to(device).eval()
        with lif_no_stats():
            lat = measure_latency(lambda X: spk(X), loader1, device, args.latency_videos)
            mem = measure_gpu_memory(lambda X: spk(X), X1, device, before=spk.reset)
        stats = measure_ops(snn, loader1, device, T, args.ops_videos)
        ops = summarize_ops(stats, T, res["T"][T]["neurons_per_step"])
        spk.reset()
        torch.cuda.empty_cache()
        return lat, mem, ops

    lat_d, mem_d, ops_d = snn_efficiency(snn_direct, r_dir)
    lat_f, mem_f, ops_f = snn_efficiency(snn_ft, r_ft)

    act_rows = [["CNN", "-", "-", "-",
                 f"{cnn_lat['mean_ms']:.1f} ± {cnn_lat['std_ms']:.1f}", fmt(cnn_mem, 0)]]
    for name, res, lat, mem in [(f"Direct SNN (T={T})", r_dir, lat_d, mem_d),
                                (f"Fine-tuned SNN (T={T})", r_ft, lat_f, mem_f)]:
        d = res["T"][T]
        act_rows.append([name, fmt(d["spikes_per_video"] / 1e6, 2),
                         fmt(d["mean_layer_rate"], 4), fmt(d["overall_rate"], 4),
                         f"{lat['mean_ms']:.1f} ± {lat['std_ms']:.1f}", fmt(mem, 0)])

    def ops_row(name, ops, snn_side=True):
        e32, e8 = ops["energy_mJ"]["fp32"], ops["energy_mJ"]["int32"]
        if not snn_side:
            return [name, fmt(ops["cnn_macs"] / 1e9), "-", "-", "-",
                    f"{e32['cnn']:.1f} (1.00x)", f"{e8['cnn']:.1f} (1.00x)"]
        return [name, "-", fmt(ops["snn_ac_ops"] / 1e9), fmt(ops["snn_mac_ops"] / 1e9),
                fmt(ops["snn_neuron_updates"] / 1e9),
                f"{e32['snn']:.1f} ({e32['ratio']:.2f}x)", f"{e8['snn']:.1f} ({e8['ratio']:.2f}x)"]

    ops_rows = [ops_row("CNN", ops_f, snn_side=False),
                ops_row(f"Direct SNN (T={T})", ops_d),
                ops_row(f"Fine-tuned SNN (T={T})", ops_f)]

    # ---------------- report ----------------
    print("\n" + "=" * 78)
    print(f"RESULTS ({args.split} split, {r_cnn['metrics'] and len(r_cnn['targets'])} videos, "
          f"single clip per video, paired={paired})")
    print("=" * 78)
    print("\nTable 1: accuracy\n")
    print(md_table(["Model", "MAE", "RMSE", "R²"], acc_rows))
    print("\nTable 2: activity and hardware measurements (batch size 1 for latency/memory)\n")
    print(md_table(["Model", "Spikes/video (M)", "Mean layer rate", "Overall rate",
                    "Latency (ms)", "Extra GPU mem (MiB)"], act_rows))
    print("\nTable 3: operations and analytical energy per video "
          "(G = 1e9; energy in mJ, ratio vs CNN)\n")
    print(md_table(["Model", "Dense MACs (G)", "SNN AC ops (G)", "SNN MAC ops (G)",
                    "Neuron updates (G)", "Energy fp32 mJ", "Energy int32 mJ"], ops_rows))
    print(f"\nWeights: {weights_mib(cnn):.1f} MiB (same for CNN and SNN). "
          f"{ops_f['n_binary_input_layers']}/{ops_f['n_layers']} layers receive binary spikes "
          f"(AC); the rest are MAC.")
    print("Latency is wall-clock on this GPU (dense kernels, no sparsity exploitation); "
          "the energy rows are analytical estimates, not measurements.")

    summary = {
        "settings": vars(args), "paired_inputs": paired, "finetuned_meta": meta,
        "accuracy": {"cnn": r_cnn["metrics"], "direct_snn": r_dir["T"][T]["metrics"],
                     "finetuned_snn": r_ft["T"][T]["metrics"]},
        "activity": {"direct": {k: v for k, v in r_dir["T"][T].items() if k != "preds"},
                     "finetuned": {k: v for k, v in r_ft["T"][T].items() if k != "preds"},
                     "layer_names": r_ft["layer_names"]},
        "latency": {"cnn": cnn_lat, "direct": lat_d, "finetuned": lat_f},
        "gpu_mem_mib": {"cnn": cnn_mem, "direct": mem_d, "finetuned": mem_f},
        "ops": {"direct": ops_d, "finetuned": ops_f},
    }
    save_json(summary, f"{args.out_prefix}_summary.json")
    np.savez(f"{args.out_prefix}_predictions.npz", targets=np.array(r_cnn["targets"]),
             cnn=np.array(r_cnn["preds"]), direct=np.array(r_dir["T"][T]["preds"]),
             finetuned=np.array(r_ft["T"][T]["preds"]))


if __name__ == "__main__":
    main()
