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
    """Upload mp4 file to S3

    Parameters
    ----------
    filepath : str
        filepath on local machine to upload to S3
    """
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


def download_video(video_title: str, video_url: str, source: str) -> None:
    """Download video from YouTube and upload to S3

    Parameters
    ----------
    video_title : str
        video title
    video_url : str
        video url
    source : str
        video source
    """
    video = yt.YouTube(video_url)
    video_length = video.length
    video_title_parsed = utils.parse_video_title(video_title)
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

        

def main(ignore_source: list[str]=None) -> None:
    for source in VIDEOS_PER_SOURCE.keys():
        if source and source in ignore_source:
            continue
        for video_title, video_url in VIDEOS_PER_SOURCE[source].items():
            tries = 0
            while tries < 10: # Workaround pytube issue
                try:
                    download_video(video_title, video_url, source)
                    break
                except Exception as e:
                    tries +=1
                    if tries==10:
                        print(f"Skipping {video_title} due to error: ", e)
                    

if __name__ == "__main__":
    main()
