import torch
import torch.nn as nn


class LIFNeuron(nn.Module):
    def __init__(self, threshold=1.0, decay=0.9):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.mem = None

    def forward(self, x):
        # 🔥 Initialize membrane on correct device & shape
        if self.mem is None or self.mem.shape != x.shape:
            self.mem = torch.zeros_like(x)

        # 🔹 Integrate (leaky accumulation)
        self.mem = self.mem * self.decay + x

        # 🔹 Generate spikes
        spike = (self.mem >= self.threshold).float()

        # 🔹 Reset only where spikes occurred
        self.mem = self.mem * (1 - spike)

        return spike

    def reset(self):
        self.mem = None