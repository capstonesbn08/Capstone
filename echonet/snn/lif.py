import torch
import torch.nn as nn


class LIFNeuron(nn.Module):
    def __init__(self, threshold=1.0, decay=0.9):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.mem = None

        # Spike rate tracking
        self.spike_count = 0
        self.element_count = 0

    def forward(self, x):
        # 🔥 Initialize membrane on correct device & shape
        if self.mem is None or self.mem.shape != x.shape:
            self.mem = torch.zeros_like(x)

        # 🔹 Integrate (leaky accumulation)
        self.mem = self.mem * self.decay + x

        # 🔹 Generate spikes
        spike = (self.mem >= self.threshold).float()

        # 🔹 Track spike rate
        self.spike_count += spike.sum().item()
        self.element_count += spike.numel()

        # 🔹 Reset only where spikes occurred
        self.mem = self.mem * (1 - spike)

        return spike

    def reset(self):
        self.mem = None

    def get_spike_rate(self):
        """Return fraction of elements that fired since last reset_stats()."""
        if self.element_count == 0:
            return 0.0
        return self.spike_count / self.element_count

    def reset_stats(self):
        """Reset spike counters (call between batches or eval runs)."""
        self.spike_count = 0
        self.element_count = 0