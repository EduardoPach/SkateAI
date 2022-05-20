import pytube as yt
from functools import lru_cache

def get_videos_url(urls: list) -> dict[str, str]:
    """Get's all URLs for each video in the playlists as well
    with the video's title.

    Parameters
    ----------
    urls : list
        A list containing URLs of playlists

    Returns
    -------
    dict[str, str]
        A dictionary with the title of the videos as keys 
        and their URLs as values.
    """
    data = {}
    for url in urls:
        playlist = yt.Playlist(url)
        for title, video_url in zip([video.title for video in playlist.videos],playlist.video_urls):
            data[title] = video_url
    return data