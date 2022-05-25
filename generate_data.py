import os
import logging

import pytube as yt
import moviepy.editor as mo
from tqdm import tqdm
import pandas as pd

import utils

def main() -> None:
    utils.initialize_data_dir()
    TRICK_CUTS = utils.get_cuts_data() # {"url": "cut_name": {"interval": [start, end], "trick_info": {...}}}
    N_CUTS = sum(len(video_cuts) for video_cuts in TRICK_CUTS.values())

    with tqdm(total=N_CUTS) as pbar:
        for url, cuts in TRICK_CUTS.items():
            video = yt.YouTube(url)
            video_title = utils.parse_video_title(video.title)
            path = f"data/videos/{video_title}"
            if not os.path.exists(path):
                os.mkdir(path)
            video.streams.filter(res="720p", file_extension='mp4', type="video", only_video=True)[0].download(output_path=path, filename="fullvideo.mp4")
            full_vid = mo.VideoFileClip(f"{path}/fullvideo.mp4")
            for cut_name, cut_info in cuts.items():
                start, end = cut_info["interval"]
                clip = full_vid.subclip(start, end)
                video_path = f"{path}/{cut_name.replace(' ', '_').lower()}.mp4"
                clip.write_videofile(video_path)
                utils.update_metadata(video_path, video_title, url, [start, end], cut_info["trick_info"])
                pbar.update(1)
            os.remove(f"{path}/fullvideo.mp4")


if __name__ == "__main__":
    main()