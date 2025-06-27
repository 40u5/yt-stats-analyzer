import json
import requests
from api_key_rotator import APIKeyRotator
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect
from typing import Optional

class YouTubeStatsCollector:

    def __init__(self):
        self.rotator = APIKeyRotator()

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _yt_get(self, url: str, params: dict, retries: int) -> dict:
        """GET with automatic key-rotation on 403 errors."""
        if retries == 0:
            return {}
        params["key"] = self.rotator.current_key()
        resp = requests.get(url, params=params)
        if resp.status_code == 403:
            self.rotator.rotate_key()
            return self._yt_get(url, params, retries - 1)

        resp.raise_for_status()
        return resp.json()

    # --------------------------------------------------------------------- #
    # Public pipeline steps
    # --------------------------------------------------------------------- #
    def fetch_channel_ids(
        self,
        max_results_per_page: int = 50,
        max_channels: int | None = None) -> list[str]:
        """
        Paginate through YouTube search results and return up to `max_channels`
        channel IDs. If max_channels is None, returns *all* channels available.
        """
        params = {
            "part": "snippet",
            "order": "date",
            "type": "video",
            "relevanceLanguage": "en",
            "regionCode": "US",
            "videoCategoryId": "28",
            "videoDuration": "medium",
            "q": "technology",
            "maxResults": str(max_results_per_page),
        }

        channels: list[str] = []
        next_page: str | None = None

        while True:
            params["maxResults"] = str(max_results_per_page)
            if next_page:
                params["pageToken"] = next_page
            else:
                params.pop("pageToken", None)

            data = self._yt_get(
                "https://www.googleapis.com/youtube/v3/search",
                params=params,
                retries=len(self.rotator.keys),
            )

            # extract this page's channel IDs
            for item in data.get("items", []):
                channels.append(item["snippet"]["channelId"])
                # if we've hit our user‐requested limit, stop here
                if max_channels and len(channels) >= max_channels:
                    return channels[:max_channels]

            # see if there's another page
            next_page = data.get("nextPageToken")
            if not next_page:
                break
        return channels

    def fetch_uploads_playlist(self, channel_id: str) -> str:
        """Return the uploads-playlist ID for a channel."""
        print(f"Fetching uploads playlist for {channel_id} …")
        data = self._yt_get(
            "https://www.googleapis.com/youtube/v3/channels",
            {"part": "contentDetails", "id": channel_id},
            retries=len(self.rotator.keys)
        )
        return data["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    def fetch_channel_video_ids(self, uploads_pl: str, max_videos: int) -> list[str]:
        """Iterate through a channel's uploads playlist (paginated)."""
        print(f"Fetching video IDs from uploads playlist {uploads_pl} …")
        video_ids: list[str] = []
        next_token = None
        batch_size = 50  # Fixed batch size for API requests

        while len(video_ids) < max_videos:
            params = {
                "part": "contentDetails",
                "playlistId": uploads_pl,
                "maxResults": batch_size,
                "pageToken": next_token or ""
            }
            data = self._yt_get(
                "https://www.googleapis.com/youtube/v3/playlistItems",
                params,
                retries=len(self.rotator.keys)
            )
            video_ids.extend(item["contentDetails"]["videoId"]
                             for item in data.get("items", []))
            next_token = data.get("nextPageToken")
            if not next_token:
                break

        return video_ids[:max_videos]

    def fetch_video_pairs(
        self, video_ids: list[str]
    ) -> Optional[list[list]]:
        """
        For each video ID, fetches title & view count.
        Detects the title's language; if not English, translates it using MarianMT.
        Returns [[translated_title, view_count], …], or None on API errors.
        """
        pairs: list[list] = []
        batch_size = 50  # Fixed batch size for API requests
        print(f"Fetching video pairs for {len(video_ids)} videos …")

        for i in range(0, len(video_ids), batch_size):
            chunk = video_ids[i : i + batch_size]
            data = self._yt_get(
                "https://www.googleapis.com/youtube/v3/videos",
                {
                    "part": "snippet,statistics",
                    "id": ",".join(chunk),
                },
                retries=len(self.rotator.keys),
            )
            if not data or "items" not in data:
                return None

            for item in data["items"]:
                raw_title = item["snippet"]["title"]
                # 1) detect language
                try:
                    lang = detect(raw_title)
                except Exception:
                    lang = "en"

                # 2) translate if needed
                if lang != "en":
                    try:
                        model_name = f"Helsinki-NLP/opus-mt-{lang}-en"
                        tokenizer = MarianTokenizer.from_pretrained(model_name)
                        model = MarianMTModel.from_pretrained(model_name)
                        inputs = tokenizer([raw_title], return_tensors="pt", padding=True)
                        translated_tokens = model.generate(**inputs)
                        title = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                    except Exception:
                        title = raw_title
                else:
                    title = raw_title

                views = int(item["statistics"].get("viewCount", 0))
                pairs.append([title, views])
                print(f"Processed video: {title} with ({views} views)")
        return pairs

    # --------------------------------------------------------------------- #
    # Orchestration + I/O
    # --------------------------------------------------------------------- #
    def build_results(self, num_channels: int, max_videos_per_channel: int) -> dict[str, list[list]]:
        """Full pipeline: discover channels → videos → `[title, views]`."""
        results: dict[str, list[list]] = {}
        channel_ids = self.fetch_channel_ids(50, num_channels)
        
        for ch_id in channel_ids:
            print(f"Processing {ch_id} …")
            uploads_pl = self.fetch_uploads_playlist(ch_id)
            vid_ids = self.fetch_channel_video_ids(uploads_pl, max_videos_per_channel)
            vid_pairs = self.fetch_video_pairs(vid_ids)
            if vid_pairs:
                results[ch_id] = vid_pairs 
        return results

    def save_to_json(self, results: dict, path: str = "results.json") -> None:
        """Dump the results as pretty-printed JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Wrote {path}")
