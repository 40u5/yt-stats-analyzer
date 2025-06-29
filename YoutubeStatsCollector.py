import json
import requests
from api_key_rotator import APIKeyRotator
from langdetect import detect_langs, LangDetectException
from googletrans import Translator
from typing import Optional
import demoji
import re

class YouTubeStatsCollector:
    def __init__(self) -> None:
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
        keywords: str,
        video_duration: str,
        order: str,
        video_category_id: Optional[int],
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
            "order": order,
            "type": "video",
            "relevanceLanguage": "en",
            "regionCode": "US",
            "videoCategoryId": "28",
            "videoDuration": video_duration,
            "q": keywords,
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
    

    def _get_language(self, text: str) -> str:
        """
        Return True if `text` is detected as English with probability ≥ threshold.
        If confidence is low, return a non-English language for translation.
        Falls back to False on errors or low confidence.
        """
        threshold = 0.95  # Set a threshold for confidence
        try:
            # get list of (lang, prob) sorted by prob descending
            langs = detect_langs(text)
            if not langs:
                return "en"
            top = langs[0]
            # If confidence is low, find a non-English language to return
            if top.prob < threshold:
                # Look for the first non-English language in the results
                for lang_obj in langs:
                    if lang_obj.lang != 'en':
                        return lang_obj.lang
                # If all detected languages are English, return the top one anyway
                return top.lang
            # High confidence case
            english_pattern = re.compile(r'^[A-Za-z0-9\s\-\.\,\!\?\:\;\'\"\/\(\)\#\@\&\*\+\=\%\$\[\]\{\}\|\\\~\`\^\<\>\_]+$')
            is_english = (top.lang == 'en' and bool(english_pattern.fullmatch(text)))
            return "en" if is_english else top.lang
        except LangDetectException as e:
            print(f"Language detection error: {e}")
            return "en"  # Default to English on detection errors

    def fetch_video_pairs(
            self, video_ids: list[str]
        ) -> Optional[list[list]]:
            """
            For each video ID, fetches title & view count.
            Detects the title's language; if not English, translates it using MarianMT.
            Returns [[translated_title, view_count], …], or None on API errors.
            """
            try:
                # Initialize list to store [title, view_count] pairs for each video
                pairs: list[list] = []
                
                # Process videos in batches to avoid API rate limits and improve efficiency
                batch_size = 50
                print(f"Fetching video pairs for {len(video_ids)} videos …")

                # Process video IDs in chunks to handle large lists efficiently
                for i in range(0, len(video_ids), batch_size):
                    # Extract current batch of video IDs
                    chunk = video_ids[i : i + batch_size]
                    
                    data = self._yt_get(
                        "https://www.googleapis.com/youtube/v3/videos",
                        {
                            "part": "snippet,statistics",
                            "id": ",".join(chunk),
                        },
                        retries=len(self.rotator.keys)
                    )

                    if not data or "items" not in data:
                        return None

                    for item in data["items"]:
                        raw_title = item["snippet"]["title"]
                        title = demoji.replace(raw_title, "")  # Remove emojis from the title
                        translator = Translator()
                        if not self._get_language(title) == "en":
                            # If the title is not in English, translate it
                            print(f"Translating title: {title}")
                            title = translator.translate(title, dest='en').text
                        # Clean the translated title to remove unwanted characters
                        cleaned_title = re.sub(r'[^A-Za-z0-9\s\-\.\,\!\?\:\;\'\"\/\(\)\#\@\&\*\+\=\%\$\[\]\{\}\|\\\~\`\^\<\>\_]+', '', title)
                        # Extract view count from statistics, defaulting to 0 if not available
                        views = int(item["statistics"].get("viewCount", 0))
                        pairs.append([cleaned_title, views])
                        print(f"Processed video: {cleaned_title} ({views} views)")
                return pairs
            except Exception as e:
                print(f"An error occurred: {e} - discarding channel")
                return None    
    # --------------------------------------------------------------------- #
    # Orchestration + I/O
    # --------------------------------------------------------------------- #


    def _validate_search_params(self, 
                               keywords: str,
                               video_duration: int,
                               order: str,
                               video_category_id: Optional[int]) -> tuple[str, str, Optional[int]]:
        """Validate search parameters and return processed values."""
        # Valid order options for YouTube Data API v3
        valid_orders = {"date", "rating", "relevance", "title", "videoCount", "viewCount"}
        if order not in valid_orders:
            raise ValueError(f"order must be one of: {', '.join(sorted(valid_orders))}")
        
        # Valid video category IDs for YouTube Data API v3
        valid_category_ids = {
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
        if video_category_id is not None and video_category_id not in valid_category_ids:
            valid_ids = sorted(valid_category_ids.keys())
            raise ValueError(f"video_category_id must be one of: {valid_ids} or None")
        
        # Map video_duration integer to YouTube API duration strings
        duration_map = ("short", "medium", "long")
        if video_duration not in [0, 1, 2]:
            raise ValueError("video_duration must be 0 (short), 1 (medium), or 2 (long)")
        
        processed_duration = duration_map[video_duration]
        return keywords, processed_duration, video_category_id





    def build_results(self,
                       num_channels: int,
                        max_videos_per_channel: int,
                        keywords: str = "technology",
                        duration_type: int = 1,
                        order: str = "relevance",
                        video_category_id: Optional[int] = None
                         ) -> dict[str, list[list]]:
        """Full pipeline: discover channels → videos → `[title, views]`."""
        duration_list = ["short", "medium", "long"]
        video_duration = duration_list[duration_type] if duration_type in (0, 1, 2) else "short"
        results: dict[str, list[list]] = {}
        channel_ids = self.fetch_channel_ids(
            keywords=keywords,
            video_duration=video_duration,
            order=order,
            video_category_id=video_category_id,
            max_results_per_page=50,
            max_channels=num_channels
        )

        for ch_id in channel_ids:
            print(f"Processing {ch_id} …")
            uploads_pl = self.fetch_uploads_playlist(ch_id)
            vid_ids = self.fetch_channel_video_ids(uploads_pl, max_videos_per_channel)
            vid_pairs = self.fetch_video_pairs(vid_ids)
            if vid_pairs:
                results[ch_id] = vid_pairs
        return results

    def save_to_json(self, results: dict, path: str = "results.json") -> None:
        """Dump the results as pretty-printed JSON, merging with existing data if file exists."""
        import os
        import json
        
        existing_data = {}
        
        # Load existing data if file exists
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing_data = {}
        
        # Check for duplicate keys (optional - depending on your needs)
        duplicate_keys = set(existing_data.keys()) & set(results.keys())
        if duplicate_keys:
            print(f"Warning: Overwriting existing keys: {duplicate_keys}")
        
        # Merge new results with existing data
        existing_data.update(results)
        
        # Write the combined data back to file (use 'w' not 'a')
        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        print(f"Wrote {path}")    

    def get_sub_counts(self, channel_ids: list[str]) -> dict[str, int]:
        """
        Fetch subscriber counts for a list of channel IDs.
        
        Args:
            channel_ids: List of YouTube channel IDs to fetch subscriber counts for.
        
        Returns:
            Dictionary mapping channel IDs to their subscriber counts.
        """
        params = {
            "part": "statistics",
            "id": ",".join(channel_ids)
        }
        data = self._yt_get(
            "https://www.googleapis.com/youtube/v3/channels",
            params,
            retries=len(self.rotator.keys)
        )
        
        sub_counts = {}
        for item in data.get("items", []):
            sub_counts[item["id"]] = int(item["statistics"].get("subscriberCount", 0))
        
        return sub_counts


# ------------------------------------------------------------------------- #
# CLI entry-point
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    fetcher = YouTubeStatsCollector()
    data = fetcher.build_results(
        num_channels=950,
        max_videos_per_channel=10,
        keywords="AI",
        duration_type=1,
        order="relevance",
        video_category_id=28
    )
    fetcher.save_to_json(data, path="results.json")