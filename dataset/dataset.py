from __future__ import annotations

import os
from typing import Union
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from sklearn.preprocessing import OrdinalEncoder

from dataset import const
from dataset import utils

class TricksDataset(Dataset):
    def __init__(
        self, 
        csv_file: Union[Path, str], 
        root_dir: Union[Path, str], 
        max_frames: Union[None, int]=None, 
        transform: Union[None, Compose]=None,
        label_enconder: OrdinalEncoder=None
    ) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.metadata = pd.read_csv(csv_file)
        self.max_frames = max_frames
        self.label_encoder = label_enconder

    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, int]]:
        video_file = self.metadata.iloc[index, 0]
        video_dir = self.metadata.iloc[index, 1]
        video_path = os.path.join(self.root_dir, video_dir, video_file)
        encoded = self.label_encoder.transform(self.metadata[const.LABELS]) if self.label_encoder else self.metadata.copy()
        labels = {col: encoded.loc[index, col] for col in const.LABELS}
        video = utils.get_video(video_path)
        video = utils.VideoToTensor()(video)

        if self.transform:
            video = self.transform(video)

        if not self.max_frames:
            return video, labels

        F, C, H, W = video.shape
        if F < self.max_frames:
            frames_diff = int(self.max_frames - F)
            padding = torch.zeros(frames_diff, C, H, W)
            video = torch.cat([padding, video])
        else:
            video = video[:self.max_frames, ...]

        return video, labels