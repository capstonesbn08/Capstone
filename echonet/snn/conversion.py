import torch
import torch.nn as nn

from echonet.snn.lif import LIFNeuron


# =========================================================
# BatchNorm Folding
# =========================================================

def _fold_bn_into_conv(conv, bn):
    """
    Fold BatchNorm parameters into the preceding convolution.
    """

    with torch.no_grad():

        scale = bn.weight / torch.sqrt(
            bn.running_var + bn.eps
        )

        if conv.weight.dim() == 5:
            scale_view = scale.view(-1, 1, 1, 1, 1)
        else:
            scale_view = scale.view(-1, 1, 1, 1)

        conv.weight.data *= scale_view

        if conv.bias is None:
            conv.bias = nn.Parameter(
                torch.zeros(
                    conv.out_channels,
                    device=conv.weight.device,
                    dtype=conv.weight.dtype,
                )
            )

        conv.bias.data = (
            (conv.bias.data - bn.running_mean) * scale
            + bn.bias
        )


def fold_batchnorm(model):
    """
    Fold BatchNorm2d/BatchNorm3d layers into preceding
    Conv2d/Conv3d layers.

    This preserves the model's inference behavior while
    removing BatchNorm from the SNN conversion path.
    """

    modules_list = list(model.named_modules())

    prev_conv = None
    pairs_to_fold = []

    for name, module in modules_list:

        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            prev_conv = module

        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):

            if prev_conv is not None:
                pairs_to_fold.append(
                    (prev_conv, module, name)
                )

            prev_conv = None

    for conv, bn, bn_name in pairs_to_fold:

        _fold_bn_into_conv(conv, bn)

        parts = bn_name.rsplit(".", 1)

        if len(parts) == 2:

            parent_name, child_name = parts

            parent = dict(
                model.named_modules()
            )[parent_name]

            setattr(
                parent,
                child_name,
                nn.Identity()
            )

        else:

            setattr(
                model,
                bn_name,
                nn.Identity()
            )

    remaining = sum(
        1
        for module in model.modules()
        if isinstance(
            module,
            (nn.BatchNorm2d, nn.BatchNorm3d)
        )
    )

    print(
        f"✅ BN folding: "
        f"{len(pairs_to_fold)} folded, "
        f"{remaining} remaining"
    )

    return model


# =========================================================
# ReLU → LIF
# =========================================================
def replace_relu_with_lif(
    model,
    threshold=1.0,
    decay=0.9,
    layer_thresholds=None,
    exclude_prefixes=None,
    surrogate=False,
    surrogate_scale=10.0,
    detach_reset=True,
    _prefix="",
):
    for name, module in model.named_children():

        full_name = f"{_prefix}.{name}" if _prefix else name

        if isinstance(module, nn.ReLU):

            if exclude_prefixes and any(
                full_name == p or full_name.startswith(p + ".")
                for p in exclude_prefixes
            ):
                continue

            layer_threshold = threshold

            if (
                layer_thresholds is not None
                and full_name in layer_thresholds
            ):
                layer_threshold = layer_thresholds[full_name]

            setattr(
                model,
                name,
                LIFNeuron(
                    threshold=layer_threshold,
                    decay=decay,
                    surrogate=surrogate,
                    surrogate_scale=surrogate_scale,
                    detach_reset=detach_reset,
                )
            )

        else:

            replace_relu_with_lif(
                module,
                threshold=threshold,
                decay=decay,
                layer_thresholds=layer_thresholds,
                exclude_prefixes=exclude_prefixes,
                surrogate=surrogate,
                surrogate_scale=surrogate_scale,
                detach_reset=detach_reset,
                _prefix=full_name,
            )

    return model


# =========================================================
# Optional RNL Scaling (legacy — kept for comparison only,
# not used by default anymore; threshold calibration via
# rnl.calibrate_thresholds() is the supported path now)
# =========================================================

def apply_weight_scaling(
    model,
    max_activations,
):
    """
    Apply conservative activation scaling.

    IMPORTANT:
    This function is intentionally conservative.

    NOTE: This matches calibration stats to layers by position
    (enumeration order of model.modules()), which is NOT
    guaranteed to match the hook-registration/execution order
    used to build max_activations. Prefer
    rnl.calibrate_thresholds() + layer_thresholds in
    convert_to_snn() instead, which matches by exact module name.
    """

    if not max_activations:
        print("⚠️ No activation statistics supplied.")
        return model

    print(
        "⚠️ Conservative RNL scaling requested."
    )

    target_modules = [
        module
        for module in model.modules()
        if isinstance(
            module,
            (nn.Conv2d, nn.Conv3d, nn.Linear)
        )
    ]

    scales = list(max_activations.values())

    applied = 0

    for i, module in enumerate(target_modules):

        if i >= len(scales):
            break

        scale = float(scales[i])

        if scale <= 0:
            continue

        # Do not modify the final regression layer.
        if (
            isinstance(module, nn.Linear)
            and module.out_features == 1
        ):
            print(
                f"⏭️ Skipping output layer {i} "
                f"(scale={scale:.4f})"
            )
            continue

        # Only use moderate scaling.
        # This prevents a huge activation outlier
        # from destroying the spike representation.
        scale = max(scale, 1.0)

        module.weight.data /= scale

        if module.bias is not None:
            module.bias.data /= scale

        applied += 1

        print(
            f"✅ Conservatively scaled layer "
            f"{i} by {scale:.4f}"
        )

    print(
        f"✅ Applied conservative scaling "
        f"to {applied} layers"
    )

    return model


# =========================================================
# Full CNN → SNN Conversion
# =========================================================
def convert_to_snn(
    model,
    layer_thresholds=None,
    lif_threshold=1.0,
    lif_decay=0.9,
    exclude_prefixes=None,
    surrogate=False,
    surrogate_scale=10.0,
    detach_reset=True,
):
    if exclude_prefixes:
        print(f"⚡ Excluding from conversion (stays analog ReLU): {exclude_prefixes}")

    if layer_thresholds:
        print(f"⚡ Using calibrated per-layer thresholds for {len(layer_thresholds)} layers (fallback={lif_threshold})")
    else:
        print(f"⚡ No calibrated thresholds supplied — using uniform threshold={lif_threshold}")

    if surrogate:
        print(f"⚡ Surrogate gradients ENABLED (scale={surrogate_scale}, detach_reset={detach_reset}) — trainable LIF layers")

    print(f"⚡ Replacing ReLU → LIF (decay={lif_decay})...")

    model = replace_relu_with_lif(
        model,
        threshold=lif_threshold,
        decay=lif_decay,
        layer_thresholds=layer_thresholds,
        exclude_prefixes=exclude_prefixes,
        surrogate=surrogate,
        surrogate_scale=surrogate_scale,
        detach_reset=detach_reset,
    )

    n_lif = sum(1 for m in model.modules() if isinstance(m, LIFNeuron))
    n_relu_remaining = sum(1 for m in model.modules() if isinstance(m, nn.ReLU))

    print(f"✅ Conversion complete — {n_lif} LIF neurons inserted, {n_relu_remaining} ReLU layers left analog")

    return model