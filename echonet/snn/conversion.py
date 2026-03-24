import torch.nn as nn
from echonet.snn.lif import LIFNeuron


# =========================================================
# 🔹 STEP 1: Replace ReLU → LIF
# =========================================================
def replace_relu_with_lif(model):
    """
    Recursively replace ReLU layers with LIF neurons
    """
    for name, module in model.named_children():

        if isinstance(module, nn.ReLU):
            setattr(model, name, LIFNeuron())

        else:
            replace_relu_with_lif(module)

    return model


# =========================================================
# 🔹 STEP 2: Apply Weight Scaling (RNL)
# =========================================================
def apply_weight_scaling(model, max_activations):
    """
    Robust scaling for deep/nested models (Conv2d + Conv3d + Linear)
    """

    scales = list(max_activations.values())
    i = 0

    for module in model.modules():

        # 🔥 FIX 1: include Conv3d
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):

            if i < len(scales):
                scale = scales[i]

                if scale > 0:
                    module.weight.data /= scale

                    if module.bias is not None:
                        module.bias.data /= scale

                    print(f"✅ Scaled layer {i} by {scale:.4f}")

                i += 1


# =========================================================
# 🔹 STEP 3: FULL CNN → SNN CONVERSION PIPELINE
# =========================================================
def convert_to_snn(model, max_activations=None):
    """
    Full conversion:
    1. Apply weight scaling (if RNL provided)
    2. Replace ReLU → LIF
    """

    if max_activations is not None:
        print("⚡ Applying weight scaling...")
        apply_weight_scaling(model, max_activations)
        print("✅ Weight scaling done")

    print("⚡ Replacing ReLU with LIF neurons...")
    model = replace_relu_with_lif(model)
    print("✅ Conversion complete")

    return model