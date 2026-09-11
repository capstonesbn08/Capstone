import torch
import torch.nn as nn

from echonet.snn.surrogate import surrogate_spike


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron.

    Inference behavior (surrogate=False, or model.eval()) is UNCHANGED
    from the original implementation: exact Heaviside spiking, hard
    reset, no gradient flow. This is what all of our conversion
    experiments used.

    Training behavior (surrogate=True and model.training): the spike
    nonlinearity is computed via SurrogateSpikeFunction, so gradients
    can flow back through the spiking dynamics for BPTT-style
    fine-tuning. The forward-pass VALUE is identical either way (real
    0/1 spikes) — only the backward pass differs.
    """

    def __init__(
        self,
        threshold=1.0,
        decay=0.9,
        surrogate=False,
        surrogate_scale=10.0,
        detach_reset=True,
    ):
        super().__init__()
        self.threshold = float(threshold)
        self.decay = float(decay)
        self.surrogate = surrogate
        self.surrogate_scale = surrogate_scale
        self.detach_reset = detach_reset

        self.mem = None
        self.spike_count = 0.0
        self.element_count = 0

    def forward(self, x):

        if (
            self.mem is None
            or self.mem.shape != x.shape
            or self.mem.device != x.device
            or self.mem.dtype != x.dtype
        ):
            self.mem = torch.zeros_like(x)

        self.mem = self.decay * self.mem + x

        use_surrogate = self.surrogate and self.training

        if use_surrogate:
            spike = surrogate_spike(
                self.mem, self.threshold, self.surrogate_scale
            )
        else:
            spike = (self.mem >= self.threshold).to(x.dtype)

        # Statistics (always computed from the real spike values,
        # detached — this must never carry gradient).
        self.spike_count += spike.detach().sum().item()
        self.element_count += spike.numel()

        if use_surrogate and self.detach_reset:
            # Standard practice (Fang et al., SpikingJelly): detach the
            # spike used for the *reset* pathway so gradient flows
            # through the spike *output* to downstream layers, but the
            # reset itself doesn't inject an extra (often destabilizing)
            # gradient term back into the membrane update.
            reset_spike = spike.detach()
        else:
            reset_spike = spike

        self.mem = self.mem * (1.0 - reset_spike)

        return spike

    def reset(self):
        self.mem = None

    def reset_stats(self):
        self.spike_count = 0.0
        self.element_count = 0

    def get_spike_rate(self):
        if self.element_count == 0:
            return 0.0
        return self.spike_count / self.element_count