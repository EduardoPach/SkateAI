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

def main(download_all: bool) -> None:
    utils.initialize_data_dir()
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


if __name__ == "__main__":
    args = parser.parse_args()
    main(download_all=args.download_all)