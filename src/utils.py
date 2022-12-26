import tqdm
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

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