from __future__ import annotations

import argparse
from typing import Callable, Union

import pandas as pd
from tqdm import tqdm

from src.pipelines import utils
from src.labeling_tool.const import VIDEOS_PER_SOURCE
from src.pipelines.const import AWS_S3_RAW_VIDEOS_DIR, LOCAL_RAW_VIDEOS_DIR

def retry(n: int, func: Callable, *args, **kwargs) -> tuple[Union[dict, None], bool]:
    for _ in range(n):
        try:
            return func(*args, **kwargs), True
        except Exception as e:
            pass
    return None, False


def main(ignore_source: list[str], store: str) -> None:
    """Pipeline that downloads raw videos from Youtube and store
    them on the adaquate location.

    Parameters
    ----------
    ignore_source : list[str], optional
        Video sources that should be ignored, by default None
    store : str, optional
        How to store the raw videos. Should be one of the options
        's3', 'local', 'both', by default 'both'
    """
    metadata = []
    total_videos = sum(
        [
            len(source) 
            for key, source in VIDEOS_PER_SOURCE.items() 
            if key not in ignore_source
        ]
    )

    progress_bar = tqdm(total=total_videos)

    for source in VIDEOS_PER_SOURCE.keys():
        if source and source in ignore_source:
            continue
        for video_title, video_url in VIDEOS_PER_SOURCE[source].items():
            progress_bar.set_description(f'Processing {video_title}')
            video_dict, success = retry(
                n=10, 
                func=utils.download_video, 
                video_title=video_title, 
                video_url=video_url, 
                source=source, 
                directory=LOCAL_RAW_VIDEOS_DIR
            )
            if not success:
                print(f"Failed to download {video_title} from {source}")
                progress_bar.update(1)
                continue
            
            if store=="local":
                progress_bar.update(1)
                continue

            remove_local = True if store=="s3" else False 

            utils.upload_mp4_to_s3(
                directory=AWS_S3_RAW_VIDEOS_DIR, 
                remove_local=remove_local, 
                **video_dict
            )

            progress_bar.update(1)
    
    progress_bar.close()
    metadata_df = pd.DataFrame(metadata)
    if store=="local":
        metadata_df.to_csv(f"{LOCAL_RAW_VIDEOS_DIR}/metadata.csv", index=False)
    elif store=="s3":
        utils.upload_df_to_s3(metadata_df, f"{AWS_S3_RAW_VIDEOS_DIR}/metadata.csv")
    else:
        metadata_df.to_csv(f"{LOCAL_RAW_VIDEOS_DIR}/metadata.csv", index=False)
        utils.upload_df_to_s3(metadata_df, f"{AWS_S3_RAW_VIDEOS_DIR}/metadata.csv")            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line downloading the data")

    parser.add_argument(
        "--ignore-source",
        nargs="+",
        type=str,
        help="Video sources that should be ignored",
        default=[]
    )

    parser.add_argument(
        "--store",
        type=str,
        help="How to store the raw videos. Should be one of the options 's3', 'local', 'both'",
        default="both"
    )

    main(**vars(parser.parse_args()))
