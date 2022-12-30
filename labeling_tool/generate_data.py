import os
import argparse

import pytube as yt
import moviepy.editor as mo

import const
import utils


parser = argparse.ArgumentParser(description="Command line downloading the data")

parser.add_argument(
    "--download-all",
    action="store_true",
    help="Wheter or not to download all videos"
)

parser.add_argument(
    "--split",
    action="store_true",
    help="Wheter or not to split dataset after download"
)

parser.add_argument(
    "--stratify-on",
    nargs="+",
    type=str,
    help="Which columns use when stratifying split",
    default=["stance", "landed"]
)

parser.add_argument(
    "--train-size",
    type=float,
    help="The train size as fraction of the total dataset i.e. between 0 and 1",
    default=0.8
)

def main(download_all: bool, split: bool, stratify_on: list, train_size: float) -> None:
    utils.initialize_data_dir(download_all)
    TRICK_CUTS = utils.get_cuts_data() # {"url": "cut_name": {"interval": [start, end], "trick_info": {...}}}
    N_CUTS = sum(len(video_cuts) for video_cuts in TRICK_CUTS.values())
    N_DOWNLOADED_VIDEOS = len(os.listdir(const.VIDEOS_DIR))
    counter = 1
    print(f"THERE ARE {N_CUTS - N_DOWNLOADED_VIDEOS} NEW CUTS TO BE DOWNLOADED\n")
    
    for url, cuts in TRICK_CUTS.items():
        video = yt.YouTube(url)
        video_title = video.title
        print("#"*100)
        print(f"DOING {video_title.upper()}\n")
        for cut_name, cut_info in cuts.items():
            video_file = f"{counter:05}.mp4"
            video_path = const.VIDEOS_DIR / video_file
            fullvideo = "fullvideo.mp4"
            fullvideo_path = const.VIDEOS_DIR/fullvideo
            if not os.path.exists(fullvideo_path):
                print("DOWNLOADING FULL VIDEO:", end=" ")
                video.streams.filter(
                    res="480p",
                    file_extension='mp4',
                    type="video",
                    only_video=True
                )[0].download(output_path=const.VIDEOS_DIR, filename=fullvideo)
                print("Success!\n")
            print("-"*100)
            print(f"DOWNLOADING {cut_name} AS {video_file} - VIDEO {counter} OUT OF {N_CUTS} ({100*counter/N_CUTS:.1f}%)\n")
            if os.path.exists(video_path) and not download_all:
                print("\nALREADY EXISTS!\n")
                print("-"*100)
                counter = counter + 1
                continue

            print("CUTTING VIDEO: ", end='')
            with mo.VideoFileClip(str(const.VIDEOS_DIR/fullvideo)) as f:
                clip = f.subclip(*cut_info["interval"])
                clip.write_videofile(str(video_path))
            
            utils.update_metadata(video_file, video_title, url, cut_info)
            print("Success!")
        
            counter = counter + 1

        os.remove(fullvideo_path)
    if not split: return
    print("#"*100)
    print(f"SPLITTING DATASET IN TRAIN AND TEST ({train_size*100:.2f} / {(1-train_size)*100:.2f}) \
        W/ STRATIFICATION ON {', '.join(stratify_on)}")
    utils.split_videos(stratify_on, train_size)

if __name__ == "__main__":
    args = parser.parse_args()
    main(
        download_all=args.download_all,
        split=args.split,
        stratify_on=args.stratify_on,
        train_size=args.train_size
    )