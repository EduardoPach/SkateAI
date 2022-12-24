import os
from typing import Union
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset import const
from dataset import utils

class TricksDataset(Dataset):
    def __init__(self, csv_file: Union[Path, str], root_dir: Union[Path, str], max_frames: Union[None, int]=None, transform: Union[None, Compose]=None) -> None:
        self.video_dir = root_dir
        self.transform = transform
        self.metadata = pd.read_csv(csv_file)
        self.max_frames = max_frames
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, np.array]:
        video_file = self.metadata.iloc[index, 0]
        video_path = os.path.join(self.video_dir, video_file)
        labels = {col: self.metadata.loc[index, col] for col in const.LABELS}
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