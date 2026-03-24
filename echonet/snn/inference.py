import torch
from echonet.snn.conversion import convert_to_snn
from echonet.snn.spiking_model import SpikingModel

def run_snn_inference(cnn_model, dataloader, device):
    cnn_model.eval()

    # Convert CNN → SNN
    snn_model = convert_to_snn(cnn_model)
    snn_model = SpikingModel(snn_model, time_steps=20)
    snn_model.to(device)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            snn_model.reset()

            pred = snn_model(x)

            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())

    return torch.cat(all_preds), torch.cat(all_targets)