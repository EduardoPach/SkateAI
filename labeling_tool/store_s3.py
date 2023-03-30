from __future__ import annotations

import os
import shutil

import tqdm
import boto3
import pytube as yt
from dotenv import load_dotenv

import utils
from const import VIDEOS_PER_SOURCE


load_dotenv()

def upload_mp4_to_s3(filepath: str, **kwargs) -> None:
    BUCKET_NAME = os.environ["S3_BUCKET"]
    s3 = boto3.client("s3")

    # Get the filename from the filepath
    filename = os.path.basename(filepath)

    # Upload the file to S3
    try:
        s3.upload_file(filepath, BUCKET_NAME, filename, ExtraArgs={"Metadata": kwargs})
        print(f"Successfully uploaded {filename} to S3 bucket {BUCKET_NAME}\n")
    except Exception as e:
        print(f"Error uploading {filename} to S3 bucket {BUCKET_NAME}: {e}\n")
    
    os.remove(filepath)
        

def main() -> None:
    for source in VIDEOS_PER_SOURCE.keys():
        for video_title, video_url in VIDEOS_PER_SOURCE[source].items():
            tries = 0
            while tries < 10: # Workaround pytube issue
                try:
                    video = yt.YouTube(video_url)
                    video_length = video.length
                    video_title_parsed = utils.parse_video_title(video_title)
                    break
                except Exception as e:
                    tries +=1
                    if tries==10:
                        raise e
                    
            video.streams.filter(
                res="480p",
                file_extension='mp4',
                type="video",
                only_video=True
            )[0].download(output_path=".", filename=f"{video_title_parsed}.mp4")

            upload_mp4_to_s3(
                f"./{video_title_parsed}.mp4",
                video_title=video_title,
                video_url=video_url,
                video_source=source,
                video_length=str(video_length)
            )

if __name__ == "__main__":
    main()
