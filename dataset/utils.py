from __future__ import annotations

from PIL import Image

import cv2
import torch
import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF


def get_video(video_file: str) -> list[Image.Image]:
    video_reader = cv2.VideoCapture(video_file)
    frames = []
    while True:
        success, frame = video_reader.read()
        if not success:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame.astype("uint8"))
        frames.append(frame)
    video_reader.release()
    return frames

class VideoToTensor:
    def __call__(self, imgs: list) -> torch.Tensor:
        return torch.stack([TF.pil_to_tensor(img) for img in imgs])