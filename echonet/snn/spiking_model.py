import torch
import torch.nn as nn


class SpikingModel(nn.Module):
    """
    Execute a converted ANN as an SNN over multiple timesteps.

    The input is already normalized by the EchoNet dataset.

    For conversion validation, the same normalized input is presented
    at every timestep. This removes Bernoulli input quantization so
    that we can measure the effect of the LIF conversion itself.
    """

    def __init__(
        self,
        model,
        time_steps=32,
        input_min=-1.0,
        input_max=5.0,
    ):
        super().__init__()

        self.model = model
        self.time_steps = int(time_steps)

        self.input_min = float(input_min)
        self.input_max = float(input_max)

    def _prepare_input(self, x):
        """
        Keep the normalized EchoNet input unchanged.

        The dataset already performs mean/std normalization.
        """

        return x.float()

    def forward(self, x):
        """
        Present the same input at every SNN timestep.

        The LIF neurons accumulate their membrane state across
        timesteps, producing a temporal spike representation.

        The final regression output is averaged across timesteps.
        """

        x = self._prepare_input(x)

        outputs = []

        # Make sure every independent forward pass starts clean.
        self.reset()

        for _ in range(self.time_steps):

            out = self.model(x)

            outputs.append(out)

        return torch.stack(outputs, dim=0).mean(dim=0)

    def reset(self):
        """
        Reset all stateful SNN neurons.
        """

        for module in self.model.modules():

            if hasattr(module, "reset"):

                module.reset()

    def reset_stats(self):
        """
        Reset spike counters in all LIF neurons.
        """

        for module in self.model.modules():

            if hasattr(module, "reset_stats"):

                module.reset_stats()

    def get_spike_rates(self):
        """
        Return spike rate for every stateful neuron.
        """

        rates = {}

        for name, module in self.model.named_modules():

            if hasattr(module, "get_spike_rate"):

                rates[name] = module.get_spike_rate()

        return rates
