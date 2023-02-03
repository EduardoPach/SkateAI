from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import wandb
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from sklearn.preprocessing import OrdinalEncoder

from dataset import TricksDataset

def train_fn(loader: DataLoader, model: nn.Module, optimizer: Optimizer, loss_fns: dict[str, nn.Module], device: "str") -> dict[str, torch.Tensor]:
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = {
            key: targets[key].float().to(device=device) \
                if 'number' in key \
                else targets[key].to(device=device) \
                for key in targets.keys()
        }

        # forward
        predictions = model(data)
        loss_dict = dict()   
        loss_total = 0
        for key in predictions.keys():
            loss_dict[key] = loss_fns[key](predictions[key], targets[key])
            loss_total += loss_dict[key]

        # backward
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        # update tqdm loop
        loop.set_postfix(loss_total=loss_total.item())
    
    loss_dict["loss_total"] = loss_total

    return {key: val.item() for key, val in loss_dict.items()}

def get_loaders(
    train_csv: Union[str, Path, pd.DataFrame],
    val_csv: Union[str, Path, pd.DataFrame],
    root_dir: Union[str, Path],
    max_frames: int,
    batch_size: int,
    train_transform: Union[Compose, None],
    val_transform: Union[Compose, None],
    num_workers: int=4,
    pin_memory: bool=True,
    label_encoder: OrdinalEncoder=None
) -> tuple[DataLoader, DataLoader]:

    train_ds = TricksDataset(
        csv_file=train_csv,
        root_dir=root_dir,
        max_frames=max_frames,
        transform=train_transform,
        label_enconder=label_encoder
    )

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = TricksDataset(
        csv_file=val_csv,
        root_dir=root_dir,
        max_frames=max_frames,
        transform=val_transform,
        label_enconder=label_encoder
    )

    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader
    

def save_checkpoint(state: dict, filename: str="my_checkpoint.pt") -> None:
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint: dict, model: nn.Module) -> None:
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def check_performance(loader: DataLoader, model: nn.Module, loss_fns: dict[str, nn.Module], device: str) -> dict[str, torch.Tensor]:
    """Calculates the average loss for an epoch of a loader

    Parameters
    ----------
    loader : DataLoader
        The loader that will be evaluated
    model : nn.Module
        Model that the performance will be checked
    loss_fns : dict[str, nn.Module]
        The losses to be used for each target variable
    device : str
        Device to use during evaluation (cuda, cpu)
    """
    model.eval()
    loss_dict = dict()
    samples = len(loader)
    avg_loss_total = 0.0
    with torch.no_grad():
        for idx, (data, target) in enumerate(loader):
            data = data.to(device)
            target = {key: target[key].to(device) for key in target.keys()}
            predictions = model(data)
            if idx==0:                
                for key in predictions.keys():
                    loss_dict[f"avg_loss_{key}"] = loss_fns[key](predictions[key], target[key]) / samples
                    avg_loss_total += loss_dict[f"avg_loss_{key}"] / samples
            else:
                for key in predictions.keys():
                    loss_dict[f"avg_loss_{key}"] += loss_fns[key](predictions[key], target[key]) / samples
                    avg_loss_total += loss_dict[f"avg_loss_{key}"] / samples
    
    loss_dict["avg_loss_total"] = avg_loss_total

    return loss_dict


def plot_frames(video: torch.Tensor, n_cols: int, **kwargs) -> None:
    F = video.shape[0]
    n_rows = int(np.ceil(66 / 5))
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, **kwargs)
    
    for idx, ax in enumerate(axes.flatten()):
        if idx+1>F: continue
        ax.imshow(video[idx].permute(1, 2, 0))
    plt.tight_layout()
    plt.show()

def wandb_log_model(model_path: Union[str, Path], name: str, type: str, **kwargs) -> None:
    model_artifact = wandb.Artifact(name=name, type=type, **kwargs)
    model_artifact.add_file(model_path)
    wandb.log_artifact(model_artifact)


def wandb_log_train_data(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    train_table = wandb.Table(dataframe=train_df)
    val_table = wandb.Table(dataframe=val_df)
    wandb.log({"train_table": train_table, "val_table": val_table})

    