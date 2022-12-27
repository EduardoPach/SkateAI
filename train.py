import yaml
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from utils import train_fn, get_loaders, load_checkpoint, save_checkpoint, check_performance
from models import ResNet18_RNN

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

EPOCHS = config["training_parameters"]["epochs"]
LEARNING_RATE = config["training_parameters"]["learning_rate"]

TRAIN_CSV = config["dataloader_parameters"]["train_csv"]
VAL_CSV = config["dataloader_parameters"]["val_csv"]
ROOT_DIR = config["dataloader_parameters"]["root_dir"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = config["dataloader_parameters"]["batch_size"]
MAX_FRAMES = config["dataloader_parameters"]["max_frames"]
NUM_WORKERS = config["dataloader_parameters"]["num_workers"]
PIN_MEMORY = config["dataloader_parameters"]["pin_memory"]

RNN_TYPE = config["model_parameters"]["rnn_type"]
RNN_LAYERS = config["model_parameters"]["rnn_layers"]
RNN_HIDDEN = config["model_parameters"]["rnn_hidden"]
TRAINABLE_BACKBONE = config["model_parameters"]["trainable_backbone"]
HEADS_PARAMS = config["model_parameters"]["heads_params"]
HEADS_PARAMS["in_features"] = RNN_HIDDEN * MAX_FRAMES

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