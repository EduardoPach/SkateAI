import torch
import torch.nn as nn
from torchvision import models

from models.utils import Identity, Heads


class ResNet18_RNN(nn.Module):
    def __init__(self, rnn_type:str, rnn_layers:int, rnn_hidden_size:int, heads_params:dict, trainable_backbone:bool) -> None:
        super(ResNet18_RNN, self).__init__()
        basemodel = models.resnet18(weights=models.resnet.ResNet18_Weights.DEFAULT)
        if not trainable_backbone:
            for param in basemodel.parameters():
                param.requires_grad = False
        
        num_features = basemodel.fc.in_features
        basemodel.fc = Identity()
        self.basemodel = basemodel
        if rnn_type=="lstm":
            self.rnn = nn.LSTM(num_features, rnn_hidden_size, rnn_layers, batch_first=True)
        elif rnn_type=="gru":
            self.rnn = nn.GRU(num_features, rnn_hidden_size, rnn_layers, batch_first=True)
        self.heads = Heads(**heads_params)

    def forward(self, x) -> list[torch.Tensor]:
        B, F, C, H, W = x.shape # Batch x Frames x Channels x Height x Width 
        x = x.view(-1, C, H, W) # Reshaping to B*F x C x H x W
        x = self.basemodel(x) # Extracting Features
        x = x.view(B, F, -1) # Reshaping to B x F x Features (ResNet18 outputs B*F x 512)
        x, _ = self.rnn(x) # output, hidden and cell states
        x = x.reshape(B, -1) # Reshaping to B x F * Hidden Size
        
        return self.heads(x)