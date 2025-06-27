import json
import requests
from api_key_rotator import APIKeyRotator
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer
from typing import Optional

class YouTubeStatsCollector:
    def __init__(self, keywords: str = "technology",
                 video_duration: int = 1,
                 order: str = "date",
                 video_category_id: Optional[int] = None) -> None:
        """
        Initialize the YouTube Stats Collector.
        
        Args:
            keywords: Search keywords for finding videos (default: "technology")
            video_duration: Video duration filter (0=short, 1=medium, 2=long) (default: 1)
            language: Language preference (default: "en")
            order: Sort order for search results (default: "date")
                   Valid options: "date", "rating", "relevance", "title", "videoCount", "viewCount"
            video_category_id: YouTube video category ID filter (default: None)
                              Valid options: 1=Film, 2=Autos, 10=Music, 15=Pets, 17=Sports, 19=Travel,
                              20=Gaming, 22=People, 23=Comedy, 24=Entertainment, 25=News, 26=Howto,
                              27=Education, 28=Science, 29=Nonprofits
        """
        self.rotator = APIKeyRotator()
        self.keywords = keywords
        
        # Valid order options for YouTube Data API v3
        self.valid_orders = {"date", "rating", "relevance", "title", "videoCount", "viewCount"}
        if order not in self.valid_orders:
            raise ValueError(f"order must be one of: {', '.join(sorted(self.valid_orders))}")
        self.order = order
        
        # Valid video category IDs for YouTube Data API v3
        self.valid_category_ids = {
            1: "Film & Animation",
            2: "Autos & Vehicles", 
            10: "Music",
            15: "Pets & Animals",
            17: "Sports",
            19: "Travel & Events",
            20: "Gaming",
            22: "People & Blogs",
            23: "Comedy",
            24: "Entertainment",
            25: "News & Politics",
            26: "Howto & Style",
            27: "Education",
            28: "Science & Technology",
            29: "Nonprofits & Activism"
        }
        if video_category_id is not None and video_category_id not in self.valid_category_ids:
            valid_ids = sorted(self.valid_category_ids.keys())
            raise ValueError(f"video_category_id must be one of: {valid_ids} or None")
        self.video_category_id = video_category_id
        
        # Map video_duration integer to YouTube API duration strings
        self.duration_map = ("short", "medium", "long")
        if video_duration not in [0, 1, 2]:
            raise ValueError("video_duration must be 0 (short), 1 (medium), or 2 (long)")
        self.video_duration = self.duration_map[video_duration]
    
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
        
        Args:
            max_results_per_page: Number of results per API call
            max_channels: Maximum number of channels to return (None for all)
            keywords: Search keywords/query
            video_duration: Duration filter (0=short <4min, 1=medium 4-20min, 2=long >20min)
            language: Language code for search results
        """
        params = {
            "part": "snippet",
            "order": self.order,
            "type": "video",
            "relevanceLanguage": "en",
            "regionCode": "US",
            "videoCategoryId": "28",
            "videoDuration": self.video_duration,
            "q": self.keywords,
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
        """Dump the results as pretty-printed JSON, appending to existing data if file exists."""
        import os
        
        existing_data = {}
        
        # Load existing data if file exists
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                # If file is corrupted or can't be read, start fresh
                existing_data = {}
        
        # Merge new results with existing data
        existing_data.update(results)
        
        # Write the combined data back to file
        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        print(f"Wrote {path}")


# ------------------------------------------------------------------------- #
# CLI entry-point
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    fetcher = YouTubeStatsCollector(video_duration=1, order="relevance", video_category_id=28, keywords="money")
    data = fetcher.build_results(num_channels=1000, max_videos_per_channel=10)
    fetcher.save_to_json(data, path="results.json")