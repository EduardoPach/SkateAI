import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from utils import train_fn, get_loaders, load_checkpoint, save_checkpoint, check_performance
from models import ResNet18_RNN



EPOCHS = None
LEARNING_RATE = 1e-4

TRAIN_CSV = "./data/metadata/metadata.csv"
VAL_CSV = None
ROOT_DIR = "./data/videos"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
MAX_FRAMES = 69
NUM_WORKERS = 4
PIN_MEMORY = True

RNN_TYPE = "lstm"
RNN_LAYERS = 2
RNN_HIDDEN = 100
TRAINABLE_BACKBONE = False
HEADS_PARAMS = {
    "in_features": RNN_HIDDEN*MAX_FRAMES,
    "byrt": [2, 2],
    "byrn": [2, 2],
    "bdrt": [2, 2],
    "bdrn": [2, 2],
    "ft": [2, 2],
    "fn": [2, 2],
    "landed": [2, 2],
    "stance":[2, 2]
}

def main():

    model = ResNet18_RNN(
        RNN_TYPE, 
        RNN_LAYERS, 
        RNN_HIDDEN, 
        HEADS_PARAMS, 
        TRAINABLE_BACKBONE
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fns = {
        'body_rotation_type': nn.CrossEntropyLoss(),
        'body_rotation_number': nn.MSELoss(),
        'board_rotation_type': nn.CrossEntropyLoss(),
        'board_rotation_number': nn.MSELoss(),
        'flip_type': nn.CrossEntropyLoss(),
        'flip_number': nn.MSELoss(),
        'landed': nn.CrossEntropyLoss(),
        'stance': nn.CrossEntropyLoss()
    }
    train_transforms = transforms.Compose([

    ])
    val_transforms = transforms.Compose([

    ])

    train_loader, val_loader = get_loaders(
        TRAIN_CSV,
        VAL_CSV,
        ROOT_DIR,
        MAX_FRAMES,
        BATCH_SIZE,
        train_transforms,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    model.train()
    model.to(DEVICE)
    for epoch in range(EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fns, DEVICE)
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

if __name__=="__main__":
    main()