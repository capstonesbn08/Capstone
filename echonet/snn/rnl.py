import torch
import torch.nn as nn
from collections import defaultdict


class RateNormalizer:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.max_activations = defaultdict(float)
        self.hooks = []

    # =========================================================
    # 🔹 Hook to capture max activation
    # =========================================================
    def _get_hook(self, name):
        def hook(module, input, output):

            # Handle tuple outputs
            if isinstance(output, tuple):
                output = output[0]

            max_val = output.detach().abs().max().item()

            # ✅ Proper accumulation
            self.max_activations[name] = max(
                self.max_activations.get(name, 0.0),
                max_val
            )

        return hook

    # =========================================================
    # 🔹 Register hooks (FIXED HERE)
    # =========================================================
    def register_hooks(self):
        """
        Attach hooks to Conv2d + Conv3d + Linear layers
        """

        for name, module in self.model.named_modules():

            clean_name = name.replace("module.", "")

            # 🔥 FINAL FIX: Added Conv3d
            if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                hook = module.register_forward_hook(
                    self._get_hook(clean_name)
                )
                self.hooks.append(hook)

    # =========================================================
    # 🔹 Remove hooks
    # =========================================================
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    # =========================================================
    # 🔹 Run calibration
    # =========================================================
    def run(self, dataloader, num_batches=10):

        self.model.eval()
        self.model.to(self.device)

        self.register_hooks()

        with torch.no_grad():
            for i, (inputs, _) in enumerate(dataloader):

                inputs = inputs.to(self.device)
                _ = self.model(inputs)

                if i >= num_batches:
                    break

        self.remove_hooks()

        print(f"✅ Collected activations from {len(self.max_activations)} layers")

        return dict(self.max_activations)


# =========================================================
# 🔹 Save / Load
# =========================================================
def save_max_activations(max_activations, path="max_activations.pth"):
    torch.save(max_activations, path)


def load_max_activations(path="max_activations.pth"):
    return torch.load(path)