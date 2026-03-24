import torch
import torch.nn as nn

from echonet.snn.conversion import convert_to_snn  # ✅ ADDED

class SpikingModel(nn.Module):
    def __init__(self, model, time_steps=20):
        super().__init__()

        # 🔥 Convert CNN → SNN
        self.model = convert_to_snn(model)   # ✅ IMPORTANT

        self.time_steps = time_steps

    def forward(self, x):
        outputs = []

        for t in range(self.time_steps):
            out = self.model(x)
            outputs.append(out)

        # average over time
        return torch.stack(outputs).mean(0)

    def reset(self):
        for m in self.model.modules():
            if hasattr(m, "reset"):
                m.reset()