import argparse
import torch

from echonet.utils.video import video_run_internal
from echonet.snn.spiking_model import SpikingModel
from echonet.snn.rnl import RateNormalizer, save_max_activations
from echonet.snn.conversion import convert_to_snn


# =========================================================
# 🔹 Parse arguments
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True)
parser.add_argument("--weights", required=True)
parser.add_argument("--device", default="cuda")

args = parser.parse_args()
device = torch.device(args.device)


# =========================================================
# 🔹 STEP 1: Load CNN
# =========================================================
model, dataloader, device = video_run_internal(
    data_dir=args.data_dir,
    weights=args.weights,
    device=device,
    return_objects=True
)

print("✅ CNN loaded")


# =========================================================
# 🔥 UNWRAP DataParallel
# =========================================================
if hasattr(model, "module"):
    print("⚡ Unwrapping DataParallel model")
    model = model.module


# =========================================================
# 🔹 STEP 2: RNL
# =========================================================
print("⚡ Running Rate Normalization (RNL)...")

rnl = RateNormalizer(model, device=device)
max_activations = rnl.run(dataloader, num_batches=20)

save_max_activations(max_activations, "max_activations.pth")

print("✅ RNL complete")
print("Collected layers:", len(max_activations))
print("Sample activations:", list(max_activations.items())[:5])


# =========================================================
# 🔹 STEP 3: CONVERT TO SNN
# =========================================================
print("⚡ Converting CNN → SNN (with scaling)...")

model = convert_to_snn(model, max_activations)

snn_model = SpikingModel(model)
snn_model.to(device)

print("✅ Converted to SNN")


# =========================================================
# 🔥 RESET FUNCTION (CRITICAL)
# =========================================================
def reset_snn(model):
    for module in model.modules():
        if hasattr(module, "reset"):
            module.reset()


# =========================================================
# 🔹 STEP 4: TEMPORAL INFERENCE
# =========================================================
snn_model.eval()

T = 30

with torch.no_grad():
    for X, y in dataloader:
        X = X.to(device)

        # 🔥 Normalize input
        # 🔥 Normalize to [0, 1] for Bernoulli
        X_min = X.min()
        X_max = X.max()
        X = (X - X_min) / (X_max - X_min + 1e-6)

        # 🔥 Clamp safety (VERY IMPORTANT)
        X = torch.clamp(X, 0.0, 1.0)

        # 🔥 Reset LIF state BEFORE sequence
        reset_snn(snn_model)

        spike_accumulator = 0

        for t in range(T):

            # 🔥 KEY FIX: stochastic spike encoding
            spike_input = torch.bernoulli(X)

            out = snn_model(spike_input)
            spike_accumulator += out

        output = spike_accumulator / T

        print("\n📊 SNN Output Stats:")
        print("Output shape:", output.shape)
        print("Output sample:", output[:5])
        print("Mean:", output.mean().item())
        print("Std:", output.std().item())

        break

print("✅ SNN inference done")