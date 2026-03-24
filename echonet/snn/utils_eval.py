import torch

def compute_metrics(preds, targets):
    mae = torch.mean(torch.abs(preds - targets))
    rmse = torch.sqrt(torch.mean((preds - targets) ** 2))

    return mae.item(), rmse.item()