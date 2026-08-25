import torch
import torch.nn as nn


class SpikingModel(nn.Module):
    def __init__(self, model, time_steps=20):
        super().__init__()

        # Model should already be converted (BN folded, weights scaled,
        # ReLU→LIF replaced) BEFORE being passed here.
        self.model = model
        self.time_steps = time_steps

    def forward(self, x):
        """
        x: input normalised to [0, 1] (NOT pre-encoded spikes).
        A fresh Bernoulli spike train is sampled each timestep.
        """
        outputs = []

        for t in range(self.time_steps):
            spike_input = torch.bernoulli(x)   # fresh sample every step
            out = self.model(spike_input)
            outputs.append(out)

        # average real-valued outputs over T timesteps
        return torch.stack(outputs).mean(0)

    def reset(self):
        for m in self.model.modules():
            if hasattr(m, "reset"):
                m.reset()