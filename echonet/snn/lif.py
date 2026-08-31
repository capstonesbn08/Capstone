import torch
import torch.nn as nn


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron.

    The neuron:
      1. Integrates incoming current into membrane potential.
      2. Emits a spike when membrane >= threshold.
      3. Resets only the spiking elements.
      4. Tracks spike statistics for evaluation.

    This implementation is stateful across timesteps.
    """

    def __init__(self, threshold=1.0, decay=0.9):
        super().__init__()

        self.threshold = float(threshold)
        self.decay = float(decay)

        self.mem = None

        self.spike_count = 0.0
        self.element_count = 0

    def forward(self, x):
        # Initialize membrane with the current tensor's shape/device.
        if (
            self.mem is None
            or self.mem.shape != x.shape
            or self.mem.device != x.device
            or self.mem.dtype != x.dtype
        ):
            self.mem = torch.zeros_like(x)

        # Leaky integration.
        self.mem = self.decay * self.mem + x

        # Generate spikes.
        spike = (self.mem >= self.threshold).to(x.dtype)

        # Statistics.
        self.spike_count += spike.detach().sum().item()
        self.element_count += spike.numel()

        # Soft reset only at locations that fired.
        self.mem = self.mem * (1.0 - spike)

        return spike

    def reset(self):
        """Reset membrane state between independent samples/batches."""
        self.mem = None

    def reset_stats(self):
        """Reset spike-rate statistics."""
        self.spike_count = 0.0
        self.element_count = 0

    def get_spike_rate(self):
        if self.element_count == 0:
            return 0.0

        return self.spike_count / self.element_count
