import torch
import torch.nn as nn
from echonet.snn.lif import LIFNeuron


# =========================================================
# 🔹 BatchNorm Folding
# =========================================================
def _fold_bn_into_conv(conv, bn):
    """Fold a BN layer's parameters into the preceding Conv layer."""
    with torch.no_grad():
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)

        if conv.weight.dim() == 5:        # Conv3d
            scale_view = scale.view(-1, 1, 1, 1, 1)
        else:                             # Conv2d
            scale_view = scale.view(-1, 1, 1, 1)

        conv.weight.data *= scale_view

        if conv.bias is None:
            conv.bias = nn.Parameter(
                torch.zeros(conv.out_channels, device=conv.weight.device)
            )
        conv.bias.data = (conv.bias.data - bn.running_mean) * scale + bn.bias


def fold_batchnorm(model):
    """
    Fold ALL BatchNorm layers into their preceding Conv layers.

    Uses module traversal order (which matches forward pass order)
    rather than direct parent-child pairs — this correctly handles
    R(2+1)D's nested Conv2Plus1D → BatchNorm3d structure.
    """
    modules_list = list(model.named_modules())

    # Track the last Conv seen in traversal order
    prev_conv = None
    pairs_to_fold = []

    for name, module in modules_list:
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            prev_conv = module
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
            if prev_conv is not None:
                pairs_to_fold.append((prev_conv, module, name))
            prev_conv = None   # consumed

    # Fold each pair
    for conv, bn, bn_name in pairs_to_fold:
        _fold_bn_into_conv(conv, bn)

        # Replace BN with Identity using parent module
        parts = bn_name.rsplit('.', 1)
        if len(parts) == 2:
            parent_name, child_name = parts
            parent = dict(model.named_modules())[parent_name]
            setattr(parent, child_name, nn.Identity())
        else:
            setattr(model, bn_name, nn.Identity())

    n_remaining = sum(1 for m in model.modules()
                      if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)))
    print(f"✅ BN folding: {len(pairs_to_fold)} folded, {n_remaining} remaining")

    return model


# =========================================================
# 🔹 Replace ReLU → LIF
# =========================================================
def replace_relu_with_lif(model, threshold=1.0):
    """Recursively replace nn.ReLU with LIFNeuron."""
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, LIFNeuron(threshold=threshold))
        else:
            replace_relu_with_lif(module, threshold=threshold)
    return model


# =========================================================
# 🔹 Apply RNL Weight Scaling
# =========================================================
def apply_weight_scaling(model, max_activations):
    """
    Scale each Conv/Linear layer's weights by its max activation.

    Output layer (out_features==1): compensated with previous scale
    so spike-rate inputs produce correct EF range.
    """
    scales = list(max_activations.values())
    target_modules = [
        m for m in model.modules()
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear))
    ]

    for i, module in enumerate(target_modules):
        if i >= len(scales):
            break
        scale = scales[i]
        if scale <= 0:
            continue

        if isinstance(module, nn.Linear) and module.out_features == 1:
            prev_scale = scales[i - 1] if i > 0 else 1.0
            module.weight.data *= prev_scale
            print(f"✅ Output layer {i}: compensated ×{prev_scale:.4f}, "
                  f"bias kept at {module.bias.data.item():.2f}")
        else:
            module.weight.data /= scale
            if module.bias is not None:
                module.bias.data /= scale
            print(f"✅ Scaled layer {i} by {scale:.4f}")


# =========================================================
# 🔹 Full CNN → SNN Conversion
# =========================================================
def convert_to_snn(model, max_activations=None, lif_threshold=1.0):
    """
    1. Apply weight scaling (if RNL activations provided)
    2. Replace ReLU → LIF neurons
    """
    if max_activations is not None:
        print("⚡ Applying weight scaling...")
        apply_weight_scaling(model, max_activations)
        print("✅ Weight scaling done")

    print(f"⚡ Replacing ReLU → LIF (threshold={lif_threshold})...")
    model = replace_relu_with_lif(model, threshold=lif_threshold)
    n_lif = sum(1 for m in model.modules() if isinstance(m, LIFNeuron))
    print(f"✅ Conversion complete — {n_lif} LIF neurons inserted")

    return model