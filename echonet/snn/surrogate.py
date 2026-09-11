import torch


class SurrogateSpikeFunction(torch.autograd.Function):
    """
    Forward pass: exact Heaviside spike (mem >= threshold -> 1, else 0),
    identical to standard inference-time LIF behavior.

    Backward pass: uses the derivative of the "fast sigmoid" surrogate
    (Zenke & Vogels style):

        S(x) = x / (1 + scale * |x|)
        dS/dx = 1 / (1 + scale * |x|)^2

    where x = mem - threshold. This gives a smooth, well-behaved
    gradient around the firing threshold instead of zero everywhere,
    which is what allows backprop to flow through spiking layers.
    """

    @staticmethod
    def forward(ctx, mem, threshold, scale):
        x = mem - threshold
        ctx.save_for_backward(x)
        ctx.scale = scale
        return (mem >= threshold).to(mem.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        scale = ctx.scale

        denom = (1.0 + scale * x.abs())
        surrogate_grad = 1.0 / (denom * denom)

        grad_mem = grad_output * surrogate_grad

        # threshold and scale are not learned parameters here
        return grad_mem, None, None


def surrogate_spike(mem, threshold, scale=10.0):
    """
    Convenience wrapper. `threshold` should be a tensor (same dtype/
    device as mem) so it participates correctly in the autograd graph
    bookkeeping, even though it currently receives no gradient.
    """
    if not torch.is_tensor(threshold):
        threshold = torch.tensor(
            float(threshold), dtype=mem.dtype, device=mem.device
        )
    return SurrogateSpikeFunction.apply(mem, threshold, scale)