from pathlib import Path
from typing import Union

import tqdm
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from dataset import TricksDataset

def train_fn(loader: DataLoader, model: nn.Module, optimizer: Optimizer, loss_fns: dict[str, nn.Module], device: "str"):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = {targets[key].to(device=device) for key in targets.keys()}

        # forward
        predictions = model(data)
        loss_dict = {}
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

def get_loaders(
    train_csv: Union[str, Path],
    val_csv: Union[str, Path, None],
    root_dir: Union[str, Path],
    max_frames: int,
    batch_size: int,
    train_transform: Compose,
    val_transform: Compose,
    num_workers: int=4,
    pin_memory: bool=True,
) -> tuple[DataLoader, DataLoader]:

    train_ds = TricksDataset(
        csv_file=train_csv,
        root_dir=root_dir,
        max_frames=max_frames,
        transform=train_transform
    )

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    if not val_csv:
        return train_loader
        
    val_ds = TricksDataset(
        csv_file=val_csv,
        root_dir=root_dir,
        max_frames=max_frames,
        transform=val_transform
    )

    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader
    

def save_checkpoint(state: dict, filename: str="my_checkpoint.pth.tar") -> None:
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint: dict, model: nn.Module) -> None:
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def check_performance(loader: DataLoader, model: nn.Module, device: str) -> None:
    model.eval()
    with torch.no_grad():
        for data, target in loader:
            data.to(device)
            target = {key: target[key].to(device) for key in target.keys()}
            predictions = model(data)