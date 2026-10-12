import argparse

from echonet.utils.snn_common import (
    add_common_args,
    get_device,
    set_seed,
    get_norm,
    get_loader,
    load_cnn,
    build_snn,
    evaluate_snn,
    md_table,
    fmt,
    save_json,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned SNN at multiple timestep counts"
    )
    add_common_args(parser)
    parser.add_argument("--finetuned_checkpoint", required=True)
    parser.add_argument("--timesteps", type=int, nargs="+",
                        default=[5, 10, 20, 50])
    args = parser.parse_args()

    if any(t < 1 for t in args.timesteps):
        parser.error("All timestep counts must be positive integers")

    device = get_device(args)
    set_seed()

    mean, std = get_norm(args)
    loader = get_loader(args, mean, std, device)

    print(f"Device: {device}")
    print(f"Split: {args.split}")
    print(f"Videos: {len(loader.dataset)}")
    print(f"Timesteps: {sorted(set(args.timesteps))}")

    cnn = load_cnn(args.weights, device)
    snn, meta = build_snn(
        cnn,
        args.lif_threshold,
        args.lif_decay,
        device,
        ckpt_path=args.finetuned_checkpoint,
    )
    print(f"Checkpoint metadata: {meta}")

    results = evaluate_snn(
        snn, loader, device, sorted(set(args.timesteps))
    )

    rows = []
    for t in sorted(set(args.timesteps)):
        m = results["T"][t]["metrics"]
        rows.append([
            t,
            fmt(m["mae"], 4),
            fmt(m["rmse"], 4),
            fmt(m["r2"], 4),
        ])

    print("\nFine-tuned SNN timestep results")
    print(md_table(["T", "MAE", "RMSE", "R²"], rows))

    output = {
        "split": args.split,
        "n_videos": results["n_videos"],
        "checkpoint": args.finetuned_checkpoint,
        "checkpoint_meta": meta,
        "timesteps": {
            str(t): {
                "metrics": results["T"][t]["metrics"],
                "spikes_per_video": results["T"][t]["spikes_per_video"],
                "mean_layer_rate": results["T"][t]["mean_layer_rate"],
                "overall_rate": results["T"][t]["overall_rate"],
            }
            for t in sorted(set(args.timesteps))
        },
    }

    filename = "snn_timestep_results.json"
    save_json(output, filename)
    print(f"\nSaved results to {filename}")


if __name__ == "__main__":
    main()
