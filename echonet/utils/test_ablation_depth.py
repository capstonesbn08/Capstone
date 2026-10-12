"""Fine-tuning depth ablation: Direct SNN / FC only / Layer4+FC / Layer3+Layer4+FC."""

import argparse
import os

from echonet.utils.snn_common import (
    add_common_args, get_device, set_seed, load_cnn, build_snn, get_norm,
    get_loader, evaluate_snn, check_same_inputs, count_trainable, md_table, fmt,
    save_json,
)


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--ckpt_fc", default=None)
    parser.add_argument("--ckpt_l4", default=None)
    parser.add_argument("--ckpt_l34", default=None)
    parser.add_argument("--T", type=int, default=50)
    args = parser.parse_args()
    args.out_prefix = args.out_prefix if args.out_prefix != "results" else "test_depth"

    device = get_device(args)
    set_seed()
    mean, std = get_norm(args)
    loader = get_loader(args, mean, std, device)
    print(f"{args.split} videos: {len(loader.dataset)}")
    cnn = load_cnn(args.weights, device)

    configs = [("Direct SNN (no fine-tuning)", None),
               ("FC only", args.ckpt_fc),
               ("Layer4 + FC", args.ckpt_l4),
               ("Layer3 + Layer4 + FC", args.ckpt_l34)]

    rows, out, checksums = [], {}, []
    for name, path in configs:
        if name.startswith("Direct"):
            snn, meta = build_snn(cnn, args.lif_threshold, args.lif_decay, device)
            trainable = 0
        else:
            if path is None or not os.path.exists(path):
                print(f"WARNING: checkpoint for '{name}' missing ({path}); skipping")
                continue
            snn, meta = build_snn(cnn, args.lif_threshold, args.lif_decay, device,
                                  ckpt_path=path)
            trainable, _ = count_trainable(snn, meta.get("freeze_prefixes"))

        print(f"\nEvaluating: {name} ...")
        res = evaluate_snn(snn, loader, device, [args.T])
        checksums.append(res["checksum"])
        d = res["T"][args.T]
        m = d["metrics"]
        rows.append([name, f"{trainable / 1e6:.2f}M" if trainable else "0",
                     meta.get("epoch", "-"), fmt(meta.get("val_r2"), 4),
                     fmt(m["mae"], 3), fmt(m["rmse"], 3), fmt(m["r2"], 4),
                     fmt(d["mean_layer_rate"], 4)])
        out[name] = {"checkpoint": path, "meta": meta, "trainable_params": trainable,
                     "metrics": m, "mean_layer_rate": d["mean_layer_rate"],
                     "overall_rate": d["overall_rate"]}

    check_same_inputs(checksums) if len(checksums) > 1 else None
    print(f"\nFine-tuning depth ablation ({args.split} split, T={args.T})\n")
    print(md_table(["Config", "Trainable params", "Ckpt epoch", "Ckpt val R²",
                    "MAE", "RMSE", "R²", "Mean layer rate"], rows))
    save_json({"settings": vars(args), "results": out}, f"{args.out_prefix}_summary.json")


if __name__ == "__main__":
    main()
