import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, params: dict) -> None:
        super(Classifier, self).__init__()

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x   