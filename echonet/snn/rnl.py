import torch
import torch.nn as nn
from collections import defaultdict


class RateNormalizer:
    """
    Collect activation statistics from a trained ANN for
    rate-based normalization before CNN -> SNN conversion.

    The normalizer records the maximum absolute activation
    produced by Conv2d, Conv3d, and Linear layers during
    calibration.
    """

    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device

        # Layer name -> maximum absolute activation observed
        self.max_activations = defaultdict(float)

        # Forward-hook handles
        self.hooks = []

    # =========================================================
    # Hook to capture maximum activation
    # =========================================================

    def _get_hook(self, name):
        def hook(module, inputs, output):

            # Some modules may return tuples
            if isinstance(output, tuple):
                output = output[0]

            if output is None:
                return

            # Maximum absolute activation for this layer
            max_val = (
                output.detach()
                .abs()
                .max()
                .item()
            )

            # Keep the maximum observed across all
            # calibration batches
            self.max_activations[name] = max(
                self.max_activations.get(name, 0.0),
                max_val,
            )

        return hook

    # =========================================================
    # Register hooks
    # =========================================================

    def register_hooks(self):
        """
        Attach forward hooks to convolutional and linear layers.

        Conv3d is important because EchoNet uses an
        R(2+1)D video network.
        """

        # Avoid registering hooks twice
        self.remove_hooks()

        for name, module in self.model.named_modules():

            clean_name = name.replace(
                "module.",
                "",
            )

            if isinstance(
                module,
                (
                    nn.Conv2d,
                    nn.Conv3d,
                    nn.Linear,
                ),
            ):

                hook = module.register_forward_hook(
                    self._get_hook(clean_name)
                )

                self.hooks.append(hook)

    # =========================================================
    # Remove hooks
    # =========================================================

    def remove_hooks(self):
        """
        Remove all registered forward hooks.
        """

        for hook in self.hooks:
            hook.remove()

        self.hooks = []

    # =========================================================
    # Calibration using a DataLoader
    # =========================================================

    def run(
        self,
        dataloader,
        num_batches=10,
    ):
        """
        Run calibration using a DataLoader.

        The DataLoader is expected to return:

            inputs, labels

        Only the inputs are used for activation calibration.
        """

        self.model.eval()
        self.model.to(self.device)

        self.register_hooks()

        try:

            with torch.no_grad():

                for i, batch in enumerate(dataloader):

                    # Support normal (inputs, labels) batches
                    if isinstance(batch, (tuple, list)):
                        inputs = batch[0]
                    else:
                        inputs = batch

                    inputs = inputs.to(
                        self.device,
                        non_blocking=True,
                    )

                    _ = self.model(inputs)

                    # Process exactly num_batches batches
                    if i + 1 >= num_batches:
                        break

        finally:

            self.remove_hooks()

        print(
            f"✅ Collected activations from "
            f"{len(self.max_activations)} layers"
        )

        return dict(self.max_activations)


# =========================================================
# Standalone calibration function (legacy — max-activation
# stats keyed by Conv/Linear name; feeds apply_weight_scaling,
# which is superseded by calibrate_thresholds below)
# =========================================================

def calibrate_model(
    model,
    calibration_batches,
):
    """
    Calibrate maximum activations from a list of tensors.

    Expected input:

        calibration_batches = [
            tensor,
            tensor,
            ...
        ]

    Each tensor should already be in the same format expected
    by the CNN, e.g.:

        (B, C, T, H, W)

    Returns:

        Dictionary mapping layer names to maximum
        absolute activation values.
    """

    model.eval()

    # Determine the model device automatically
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    normalizer = RateNormalizer(
        model=model,
        device=device,
    )

    normalizer.register_hooks()

    try:

        with torch.no_grad():

            for inputs in calibration_batches:

                inputs = inputs.to(
                    device,
                    non_blocking=True,
                )

                _ = model(inputs)

    finally:

        normalizer.remove_hooks()

    print(
        f"RNL collected "
        f"{len(normalizer.max_activations)} layers"
    )

    return dict(
        normalizer.max_activations
    )


# =========================================================
# Per-layer LIF threshold calibration (supported path)
# =========================================================

def calibrate_thresholds(
    model,
    calibration_batches,
    percentile=99.9,
    min_threshold=1e-3,
    anchor_threshold=0.5,
):
    """
    Calibrate a per-ReLU-layer LIF threshold from activation
    statistics, keyed by the exact dotted module name so it
    lines up 1:1 with the names replace_relu_with_lif() uses
    when it walks the model and performs the replacement.

    Raw per-layer percentile activations are then rescaled so
    their median equals anchor_threshold. This preserves the
    relative differences between layers (deep layers get lower
    thresholds than shallow ones) while keeping the absolute
    scale in the regime that this LIF's decay/reset dynamics
    are known to work in.
    """

    model.eval()

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    stats = defaultdict(list)
    hooks = []

    def make_hook(name):
        def hook(module, inputs, output):

            if isinstance(output, tuple):
                output = output[0]

            if output is None:
                return

            with torch.no_grad():

                flat = output.detach().float().reshape(-1)

                if flat.numel() > 5_000_000:
                    idx = torch.randperm(
                        flat.numel(),
                        device=flat.device,
                    )[:5_000_000]
                    flat = flat[idx]

                q = torch.quantile(
                    flat,
                    percentile / 100.0,
                )

                stats[name].append(q.item())

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            hooks.append(
                module.register_forward_hook(make_hook(name))
            )

    try:

        with torch.no_grad():

            for inputs in calibration_batches:

                inputs = inputs.to(
                    device,
                    non_blocking=True,
                )

                _ = model(inputs)

    finally:

        for h in hooks:
            h.remove()

    thresholds = {
        name: max(sum(vals) / len(vals), min_threshold)
        for name, vals in stats.items()
    }

    if anchor_threshold is not None and thresholds:

        raw_values = sorted(thresholds.values())
        mid = len(raw_values) // 2
        if len(raw_values) % 2 == 0:
            raw_median = (raw_values[mid - 1] + raw_values[mid]) / 2.0
        else:
            raw_median = raw_values[mid]

        if raw_median > 0:
            scale_factor = anchor_threshold / raw_median
            thresholds = {
                name: max(val * scale_factor, min_threshold)
                for name, val in thresholds.items()
            }

            print(
                f"⚖️  Anchored thresholds: raw median={raw_median:.4f} "
                f"-> target={anchor_threshold:.4f} "
                f"(scale factor={scale_factor:.4f})"
            )

    print(
        f"✅ Calibrated thresholds for "
        f"{len(thresholds)} ReLU layers "
        f"(p{percentile}"
        + (
            f", anchored to {anchor_threshold}"
            if anchor_threshold is not None
            else ""
        )
        + ")"
    )

    return thresholds


# =========================================================
# Save activation statistics
# =========================================================

def save_max_activations(
    max_activations,
    path="max_activations.pth",
):
    """
    Save calibrated activation statistics.
    """

    torch.save(
        max_activations,
        path,
    )

    print(
        f"✅ Saved activation statistics to: {path}"
    )


# =========================================================
# Load activation statistics
# =========================================================

def load_max_activations(
    path="max_activations.pth",
):
    """
    Load previously calibrated activation statistics.
    """

    max_activations = torch.load(
        path,
        map_location="cpu",
    )

    print(
        f"✅ Loaded activation statistics from: {path}"
    )

    return max_activations
