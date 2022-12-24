import tqdm
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

def train_fn(loader: DataLoader, model: nn.Module, optimizer: Optimizer, loss_fns: list[nn.Module], device: "str"):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().unsqueeze(1).to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())