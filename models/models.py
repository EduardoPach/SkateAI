import torch
import torch.nn as nn
from torchvision import models

from models.utils import Identity, Classifier


class ResNet18_RNN(nn.Module):
    def __init__(self, rnn_type:str, rnn_layers:int, rnn_hidden_size:int, classifier_params:dict, trainable_backbone:bool) -> None:
        super(ResNet18_RNN, self).__init__()
        basemodel = models.resnet18(weights=models.resnet.ResNet18_Weights.DEFAULT)
        if not trainable_backbone:
            for param in basemodel.parameters():
                param.requires_grad = False
        
        num_features = basemodel.fc.in_features
        basemodel.fc = Identity()
        self.basemodel = basemodel
        if rnn_type=="lstm":
            self.rnn = nn.LSTM(num_features, rnn_hidden_size, rnn_layers)
        else:
            self.rnn = nn.GRU(num_features, rnn_hidden_size, rnn_layers)
        self.classifier = Classifier(classifier_params)

    def forward(self, x):
        # Batch x Frames x Channels x Height x Width 
        B, F, C, H, W = x.shape
        # First pass
        features = self.basemodel(x[:,0]) # extracted features for basemodel
        output, (h, c) = self.rnn(features.unsqueeze(1)) # output, hidden and cell states
        # Looping the other frames
        for i in range(1, F):
            features = self.basemodel(x[:,i]) # extracted features for basemodel
            output, (h, c) = self.rnn(features.unsqueeze(1), (h, c)) # output, hidden and cell states
        output = self.classifier()