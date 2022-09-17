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
            print("-"*50)
            print(f"DOING {video_title.upper()}")
            path = f"data/videos/{video_title}"
            if not os.path.exists(path):
                os.mkdir(path)
            if len(os.listdir(path))==len(cuts.keys()):
                pbar.update(1)
                print(f"{video_title.upper()} WAS ALREADY DONE!")
                continue

            for cut_name, cut_info in cuts.items():
                if not os.path.exists(f"{path}/fullvideo.mp4"):
                    print("DOWNLOADING FULL VIDEO: ", end=" ")
                    video.streams.filter(res="720p", file_extension='mp4', type="video", only_video=True)[0].download(output_path=path, filename="fullvideo.mp4")
                    print("Success!")

                print(f"STARTING {cut_name}")
                start, end = cut_info["interval"]
                video_path = f"{path}/{cut_name.replace(' ', '_').lower()}.mp4" 
                if os.path.exists(video_path):
                    print("ALREADY EXISTS!\n")
                else:
                    print("CUTTING VIDEO: ", end='')
                    with mo.VideoFileClip(f"{path}/fullvideo.mp4") as f:
                        clip = f.subclip(start, end)
                        clip.write_videofile(video_path)
                    print("Success!")
                    
                    utils.update_metadata(video_path, video_title, url, [start, end], cut_info["trick_info"])
                pbar.update(1)
                print("")
            os.remove(f"{path}/fullvideo.mp4")
            print("-"*50)


if __name__ == "__main__":
    main()