import cv2
import numpy as np
import pandas as pd


def get_video(video_file: str) -> np.array:
    video_reader = cv2.VideoCapture(video_file)
    frames = []
    while True:
        success, frame = video_reader.read()
        if not success:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    video_reader.release()
    return np.array(frames)