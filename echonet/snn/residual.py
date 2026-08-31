import torch
import torch.nn as nn
from collections import defaultdict


class ScaleModule(nn.Module):
    """
    Multiplies input by a fixed scalar. Inserted as a BasicBlock's
    `downsample` so the residual (skip) branch gets rescaled before
    the residual add, without needing to touch BasicBlock.forward()
    at all — torchvision's forward() already routes x through
    self.downsample whenever it is not None.
    """

    def __init__(self, scale=1.0):
        super().__init__()
        self.register_buffer("scale", torch.tensor(float(scale)))

    def forward(self, x):
        return x * self.scale


def find_basic_blocks(model):
    """
    Locate BasicBlock instances in a torchvision video ResNet
    (e.g. r2plus1d_18). Returns a list of (dotted_name, module).
    """
    from torchvision.models.video.resnet import BasicBlock

    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, BasicBlock)
    ]


def calibrate_residual_balance(model, calibration_batches):
    """
    For each BasicBlock, measure the typical magnitude of the
    conv-path branch (conv1->conv2 output, pre-add) and the
    residual/skip branch (downsample(x), or x itself when there
    is no downsample), both pre-add. Returns a scale factor per
    block: how much to multiply the residual branch by so both
    branches contribute comparably to the sum.

    Must run BEFORE ReLU->LIF conversion, on the BN-folded model,
    so both branches are still real (non-spiking) float tensors —
    this measures the mismatch that spiking dynamics will later
    have to route through unchanged unless corrected.
    """

    blocks = find_basic_blocks(model)

    if not blocks:
        print("⚠️  No BasicBlock modules found — nothing to balance.")
        return {}

    conv_stats = defaultdict(list)
    residual_stats = defaultdict(list)
    hooks = []

    def conv_hook(name):
        def hook(module, inputs, output):
            with torch.no_grad():
                conv_stats[name].append(
                    output.detach().float().abs().mean().item()
                )
        return hook

    def residual_pre_hook(name):
        # Fires before the block's forward() runs, so inputs[0] is
        # exactly the x that forward() will use as `residual = x`.
        def hook(module, inputs):
            with torch.no_grad():
                residual_stats[name].append(
                    inputs[0].detach().float().abs().mean().item()
                )
        return hook

    def downsample_hook(name):
        def hook(module, inputs, output):
            with torch.no_grad():
                residual_stats[name].append(
                    output.detach().float().abs().mean().item()
                )
        return hook

    for name, block in blocks:
        hooks.append(block.conv2.register_forward_hook(conv_hook(name)))
        if block.downsample is not None:
            hooks.append(
                block.downsample.register_forward_hook(downsample_hook(name))
            )
        else:
            hooks.append(
                block.register_forward_pre_hook(residual_pre_hook(name))
            )

    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    try:
        with torch.no_grad():
            for inputs in calibration_batches:
                inputs = inputs.to(device, non_blocking=True)
                _ = model(inputs)
    finally:
        for h in hooks:
            h.remove()

    scales = {}

    for name, _ in blocks:
        conv_vals = conv_stats.get(name, [])
        res_vals = residual_stats.get(name, [])

        if not conv_vals or not res_vals:
            continue

        conv_avg = sum(conv_vals) / len(conv_vals)
        res_avg = sum(res_vals) / len(res_vals)

        if res_avg <= 1e-8:
            continue

        scale = conv_avg / res_avg
        scales[name] = scale

        print(
            f"  {name:<28} conv_avg={conv_avg:.4f}  "
            f"res_avg={res_avg:.4f}  scale={scale:.4f}"
        )

    print(f"✅ Computed residual-branch balance for {len(scales)} blocks")

    return scales


def apply_residual_balance(model, scales):
    """
    Apply calibrated residual-branch scale factors to each
    BasicBlock, by rescaling its existing downsample conv's
    weight/bias, or inserting a ScaleModule as the downsample
    when none existed. Must be called on the BN-folded model,
    BEFORE ReLU->LIF conversion.

    IMPORTANT: this mutates the model it's given in place. Call
    it on the SNN copy (snn_cnn), never on the original CNN used
    for the CNN baseline evaluation.
    """

    if not scales:
        print("⚠️  No residual balance scales supplied — skipping.")
        return model

    blocks = dict(find_basic_blocks(model))
    applied = 0

    for name, scale in scales.items():

        block = blocks.get(name)
        if block is None:
            continue

        if block.downsample is not None:

            found_conv = False
            for m in block.downsample.modules():
                if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                    with torch.no_grad():
                        m.weight.data *= scale
                        if m.bias is not None:
                            m.bias.data *= scale
                    found_conv = True

            if not found_conv:
                block.downsample = nn.Sequential(
                    block.downsample,
                    ScaleModule(scale),
                )

        else:

            block.downsample = ScaleModule(scale)

        applied += 1

    print(
        f"✅ Applied residual-branch balance to "
        f"{applied}/{len(scales)} blocks"
    )

    return model