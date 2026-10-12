"""Shared helpers for the CNN-vs-SNN comparison and ablation scripts."""

import contextlib
import copy
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision

import echonet
from echonet.snn.conversion import fold_batchnorm, convert_to_snn
from echonet.snn.lif import LIFNeuron

SEED = 42

# Energy per operation in pJ (45 nm CMOS, Horowitz ISSCC 2014, as commonly
# quoted in SNN papers). Verify against the paper before citing.
ENERGY_SETS = {
    "fp32": {"e_mac": 4.6, "e_ac": 0.9},
    "int32": {"e_mac": 3.2, "e_ac": 0.1},
}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_common_args(parser):
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--weights", required=True, help="Original CNN checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Smoke test: use only the first N videos")
    parser.add_argument("--lif_threshold", type=float, default=0.5)
    parser.add_argument("--lif_decay", type=float, default=0.9)
    parser.add_argument("--mean", type=float, nargs=3, default=None,
                        help="3 channel means (copy from the training log)")
    parser.add_argument("--std", type=float, nargs=3, default=None,
                        help="3 channel stds (copy from the training log)")
    parser.add_argument("--out_prefix", default="results")


def get_device(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA unavailable, falling back to CPU")
        return torch.device("cpu")
    return torch.device(args.device)


def load_cnn(weights, device):
    ckpt = torch.load(weights, map_location=device)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sd = {(k[len("module."):] if k.startswith("module.") else k): v
          for k, v in sd.items()}
    model = torchvision.models.video.r2plus1d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"WARNING: CNN load: {len(missing)} missing, {len(unexpected)} unexpected keys")
    return model.to(device).eval()


def build_snn(cnn, lif_threshold, lif_decay, device, ckpt_path=None):
    """BN-fold + ReLU->LIF on a copy of the CNN; optionally load fine-tuned weights."""
    snn = copy.deepcopy(cnn).to(device)
    snn = fold_batchnorm(snn)
    snn = convert_to_snn(snn, layer_thresholds=None, lif_threshold=lif_threshold,
                         lif_decay=lif_decay, surrogate=False)
    meta = {}
    if ckpt_path is not None:
        ck = torch.load(ckpt_path, map_location=device)
        snn.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck, strict=True)
        meta = {k: ck.get(k) for k in ("epoch", "val_r2", "val_mae", "val_rmse")}
        meta["freeze_prefixes"] = (ck.get("settings") or {}).get("freeze_prefixes")
    return snn.eval(), meta


def count_trainable(model, freeze_prefixes):
    prefixes = [p.strip() for p in (freeze_prefixes or "").split(",") if p.strip()]
    trainable = total = 0
    for name, p in model.named_parameters():
        total += p.numel()
        if not any(name == q or name.startswith(q + ".") for q in prefixes):
            trainable += p.numel()
    return trainable, total


def get_norm(args):
    if args.mean and args.std:
        print("Using mean/std supplied on the command line")
        return (np.array(args.mean, dtype=np.float32),
                np.array(args.std, dtype=np.float32))
    print("WARNING: --mean/--std not given; resampling from train (results will drift slightly)")
    return echonet.utils.get_mean_and_std(
        echonet.datasets.Echo(root=args.data_dir, split="train"))


def get_loader(args, mean, std, device, batch_size=None, limit=None):
    ds = echonet.datasets.Echo(root=args.data_dir, split=args.split, target_type="EF",
                               mean=mean, std=std, length=32, period=2)
    cap = len(ds)
    if args.max_samples:
        cap = min(cap, args.max_samples)
    if limit:
        cap = min(cap, limit)
    if cap < len(ds):
        ds = torch.utils.data.Subset(ds, range(cap))
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size or args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"))


def regression_metrics(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    err = pred - target
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    return {"mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")}


def check_same_inputs(checksums):
    ref = checksums[0]
    ok = all(abs(c - ref) <= 1e-6 * max(abs(ref), 1.0) for c in checksums)
    print("Identical test clips across all evaluated models: "
          + ("YES" if ok else "NO  <-- comparison is NOT paired, see notes"))
    return ok


# ------------------------------------------------------------------ accuracy

@torch.no_grad()
def evaluate_cnn(cnn, loader, device):
    set_seed()
    preds, targets, checksum = [], [], 0.0
    for X, y in loader:
        X = X.to(device, non_blocking=True).float()
        checksum += float(X.double().sum())
        preds += cnn(X).reshape(-1).cpu().tolist()
        targets += y.reshape(-1).tolist()
    return {"preds": preds, "targets": targets, "checksum": checksum,
            "metrics": regression_metrics(preds, targets)}


def get_lifs(model):
    return [m for m in model.modules() if isinstance(m, LIFNeuron)]


def reset_lifs(model, stats=False):
    for m in get_lifs(model):
        m.reset()
        if stats:
            m.reset_stats()


@torch.no_grad()
def evaluate_snn(snn, loader, device, T_list):
    """
    One pass at max(T_list). The prediction at T=k is the running mean of the
    first k timestep outputs, which is exactly what a separate T=k run
    produces (the computation is causal and the input is identical every step).
    Spike statistics are snapshotted at each k the same way.
    """
    T_list = sorted(set(int(t) for t in T_list))
    lif_names = [n for n, m in snn.named_modules() if isinstance(m, LIFNeuron)]
    lifs = get_lifs(snn)
    L = len(lifs)
    preds = {T: [] for T in T_list}
    sp = {T: np.zeros(L) for T in T_list}
    el = {T: np.zeros(L) for T in T_list}
    targets, checksum = [], 0.0

    set_seed()
    for X, y in loader:
        X = X.to(device, non_blocking=True).float()
        checksum += float(X.double().sum())
        targets += y.reshape(-1).tolist()
        reset_lifs(snn, stats=True)
        running = None
        for t in range(1, T_list[-1] + 1):
            out = snn(X).reshape(-1)
            running = out if running is None else running + out
            if t in preds:
                preds[t] += (running / t).cpu().tolist()
                for i, m in enumerate(lifs):
                    sp[t][i] += m.spike_count
                    el[t][i] += m.element_count

    reset_lifs(snn)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    n = len(targets)
    result = {"targets": targets, "checksum": checksum, "n_videos": n,
              "layer_names": lif_names, "T": {}}
    for T in T_list:
        rates = sp[T] / np.maximum(el[T], 1)
        result["T"][T] = {
            "preds": preds[T],
            "metrics": regression_metrics(preds[T], targets),
            "layer_rates": rates.tolist(),
            "mean_layer_rate": float(rates.mean()),
            "overall_rate": float(sp[T].sum() / max(el[T].sum(), 1)),
            "spikes_per_video": float(sp[T].sum() / n),
            "neurons_per_step": float(el[T].sum() / (n * T)),
        }
    return result


# ---------------------------------------------------------------- efficiency

@contextlib.contextmanager
def lif_no_stats():
    """Temporarily swap LIF.forward for a version without spike bookkeeping
    (the .item() calls force GPU syncs and would inflate latency).
    Valid in eval mode: identical spiking behaviour."""
    original = LIFNeuron.forward

    def forward(self, x):
        if (self.mem is None or self.mem.shape != x.shape
                or self.mem.device != x.device or self.mem.dtype != x.dtype):
            self.mem = torch.zeros_like(x)
        self.mem = self.decay * self.mem + x
        spike = (self.mem >= self.threshold).to(x.dtype)
        self.mem = self.mem * (1.0 - spike)
        return spike

    LIFNeuron.forward = forward
    try:
        yield
    finally:
        LIFNeuron.forward = original


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def measure_latency(fn, loader, device, n_videos, warmup=3):
    """Per-video latency in ms. Loader must have batch_size=1."""
    times = []
    for i, (X, _) in enumerate(loader):
        if i >= n_videos + warmup:
            break
        X = X.to(device).float()
        _sync(device)
        t0 = time.perf_counter()
        fn(X)
        _sync(device)
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1000.0)
    return {"mean_ms": float(np.mean(times)), "std_ms": float(np.std(times)),
            "median_ms": float(np.median(times)), "n": len(times)}


@torch.no_grad()
def measure_gpu_memory(fn, X, device, before=None):
    """Peak extra GPU memory (MiB) during one forward pass, beyond what was
    already resident (weights etc.). For the SNN this includes membrane state."""
    if device.type != "cuda":
        return None
    if before is not None:
        before()
    _sync(device)
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn(X)
    _sync(device)
    return (torch.cuda.max_memory_allocated() - base) / 2 ** 20


def weights_mib(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 2 ** 20


def layer_macs(mod, out):
    if isinstance(mod, nn.Linear):
        return mod.in_features * mod.out_features
    per_sample_out = out.numel() // out.shape[0]
    return per_sample_out * int(np.prod(mod.kernel_size)) * (mod.in_channels // mod.groups)


@torch.no_grad()
def measure_ops(snn, loader, device, T, n_videos):
    """Run the SNN for T steps on a few videos; for every Conv/Linear layer record
    dense MACs, the measured fraction of non-zero input elements, and whether the
    input was strictly binary (spikes -> accumulate-only, AC) or not (real-valued
    -> full MAC). Use batch_size=1."""
    layers = [(n, m) for n, m in snn.named_modules()
              if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear))]
    stats = {n: {"macs": None, "nz": 0.0, "el": 0.0, "binary": True} for n, _ in layers}
    hooks = []

    def make_hook(name):
        def hook(mod, inp, out):
            x = inp[0]
            s = stats[name]
            if s["macs"] is None:
                s["macs"] = layer_macs(mod, out)
            s["nz"] += float((x != 0).sum())
            s["el"] += float(x.numel())
            if s["binary"] and not bool(((x == 0) | (x == 1)).all()):
                s["binary"] = False
        return hook

    for n, m in layers:
        hooks.append(m.register_forward_hook(make_hook(n)))
    try:
        seen = 0
        for X, _ in loader:
            X = X.to(device).float()
            reset_lifs(snn)
            for _ in range(T):
                snn(X)
            seen += X.shape[0]
            if seen >= n_videos:
                break
    finally:
        for h in hooks:
            h.remove()
        reset_lifs(snn)
    return stats


def summarize_ops(stats, T, neurons_per_step):
    """Per-video operation counts and energy (mJ) for CNN vs SNN over T steps."""
    dense_macs = float(sum(s["macs"] for s in stats.values()))
    ac_ops = mac_ops = 0.0
    for s in stats.values():
        density = s["nz"] / max(s["el"], 1.0)
        if s["binary"]:
            ac_ops += T * s["macs"] * density
        else:
            mac_ops += T * s["macs"]
    neuron_updates = T * neurons_per_step
    out = {"cnn_macs": dense_macs, "snn_ac_ops": ac_ops, "snn_mac_ops": mac_ops,
           "snn_neuron_updates": neuron_updates,
           "n_binary_input_layers": sum(1 for s in stats.values() if s["binary"]),
           "n_layers": len(stats), "energy_mJ": {}}
    for name, e in ENERGY_SETS.items():
        cnn_e = dense_macs * e["e_mac"]
        syn_e = mac_ops * e["e_mac"] + ac_ops * e["e_ac"]
        snn_e = syn_e + neuron_updates * e["e_mac"]   # one membrane update ~ 1 MAC
        out["energy_mJ"][name] = {"cnn": cnn_e / 1e9, "snn": snn_e / 1e9,
                                  "snn_synaptic_only": syn_e / 1e9,
                                  "ratio": snn_e / cnn_e}
    return out


# ------------------------------------------------------------------- output

def fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"Saved: {path}")
