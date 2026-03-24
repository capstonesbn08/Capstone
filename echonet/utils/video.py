"""Functions for training and running EF prediction."""

import math
import os
import time

import click
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
import torchvision
import tqdm

import echonet


# =========================
# 🔹 INTERNAL FUNCTION (NO CLICK)
# =========================
def video_run_internal(
    data_dir=None,
    output=None,
    task="EF",
    model_name="r2plus1d_18",
    pretrained=True,
    weights=None,
    run_test=False,
    num_epochs=45,
    lr=1e-4,
    weight_decay=1e-4,
    lr_step_period=15,
    frames=32,
    period=2,
    num_train_patients=None,
    num_workers=0,
    batch_size=20,
    device=None,
    seed=0,
    return_objects=False,
):

    np.random.seed(seed)
    torch.manual_seed(seed)

    if output is None:
        output = os.path.join(
            "output", "video",
            "{}_{}_{}_{}".format(
                model_name, frames, period,
                "pretrained" if pretrained else "random"
            )
        )
    os.makedirs(output, exist_ok=True)

    # 🔥 FIX: ensure device is torch.device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # 🔹 MODEL
    model = torchvision.models.video.__dict__[model_name](pretrained=pretrained)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.fc.bias.data[0] = 55.6

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)

    model.to(device)

    if weights is not None:
        checkpoint = torch.load(weights, map_location=device)

        state_dict = checkpoint['state_dict']

        # 🔥 Remove "module." prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        new_state_dict = {f"module.{k}": v for k, v in new_state_dict.items()}
        model.load_state_dict(new_state_dict)

    # 🔹 OPTIMIZER
    optim = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay
    )

    if lr_step_period is None:
        lr_step_period = math.inf

    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    # 🔹 DATA
    mean, std = echonet.utils.get_mean_and_std(
        echonet.datasets.Echo(root=data_dir, split="train")
    )

    kwargs = {
        "target_type": task,
        "mean": mean,
        "std": std,
        "length": frames,
        "period": period,
    }

    dataset = {}
    dataset["train"] = echonet.datasets.Echo(
        root=data_dir, split="train", **kwargs, pad=12
    )

    if num_train_patients is not None and len(dataset["train"]) > num_train_patients:
        indices = np.random.choice(len(dataset["train"]), num_train_patients, replace=False)
        dataset["train"] = torch.utils.data.Subset(dataset["train"], indices)

    dataset["val"] = echonet.datasets.Echo(
        root=data_dir, split="val", **kwargs
    )

    # ✅ EARLY RETURN FOR SNN
    if return_objects:
        val_loader = torch.utils.data.DataLoader(
            dataset["val"],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=(device.type == "cuda")
        )
        return model, val_loader, device

    # 🔹 TRAIN LOOP
    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")

        try:
            checkpoint = torch.load(
                os.path.join(output, "checkpoint.pt"),
                map_location=device
            )
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
        except FileNotFoundError:
            pass

        for epoch in range(epoch_resume, num_epochs):
            print(f"Epoch #{epoch}")

            for phase in ['train', 'val']:
                ds = dataset[phase]

                dataloader = torch.utils.data.DataLoader(
                    ds,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=True,
                    pin_memory=(device.type == "cuda"),
                    drop_last=(phase == "train")
                )

                loss, yhat, y = run_epoch(
                    model, dataloader,
                    phase == "train", optim, device
                )

            scheduler.step()

            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_loss': bestLoss,
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
            }

            torch.save(save, os.path.join(output, "checkpoint.pt"))

            if loss < bestLoss:
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = loss


# =========================
# 🔹 CLI WRAPPER (CLICK)
# =========================
@click.command("video")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--output", type=click.Path(file_okay=False), default=None)
@click.option("--task", default="EF")
@click.option("--model_name", default="r2plus1d_18")
@click.option("--pretrained/--random", default=True)
@click.option("--weights", type=click.Path(exists=True), default=None)
@click.option("--run_test/--skip_test", default=False)
@click.option("--num_epochs", default=45)
@click.option("--lr", default=1e-4)
@click.option("--weight_decay", default=1e-4)
@click.option("--lr_step_period", default=15)
@click.option("--frames", default=32)
@click.option("--period", default=2)
@click.option("--num_train_patients", default=None)
@click.option("--num_workers", default=4)
@click.option("--batch_size", default=20)
@click.option("--device", default=None)
@click.option("--seed", default=0)
def run(**kwargs):
    video_run_internal(**kwargs)


# =========================
# 🔹 RUN EPOCH
# =========================
def run_epoch(model, dataloader, train, optim, device, save_all=False, block_size=None):

    model.train(train)

    total = 0
    n = 0

    yhat = []
    y = []

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (X, outcome) in dataloader:

                y.append(outcome.detach().cpu().numpy())

                X = X.to(device)
                outcome = outcome.to(device)

                average = (len(X.shape) == 6)
                if average:
                    batch, n_clips, c, f, h, w = X.shape
                    X = X.view(-1, c, f, h, w)

                if block_size is None:
                    outputs = model(X)
                else:
                    outputs = torch.cat([
                        model(X[j:(j + block_size), ...])
                        for j in range(0, X.shape[0], block_size)
                    ])

                if average:
                    outputs = outputs.view(batch, n_clips, -1).mean(1)

                yhat.append(outputs.detach().cpu().numpy())

                loss = torch.nn.functional.mse_loss(
                    outputs.view(-1), outcome
                )

                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                total += loss.item() * X.size(0)
                n += X.size(0)

                pbar.update()

    yhat = np.concatenate(yhat)
    y = np.concatenate(y)

    return total / max(n, 1), yhat, y