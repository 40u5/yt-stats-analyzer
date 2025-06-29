"""
YouTube Statistics Collector

A tool for collecting YouTube video statistics and metadata from channels
based on search criteria. Supports API key rotation and quota management.
"""

import json
import os
import re
from typing import Optional, Dict, List, Tuple

import demoji
import requests
from googletrans import Translator
from langdetect import detect_langs, LangDetectException

from api_key_rotator import APIKeyRotator
from exceptions import QuotaExhaustedError


# Constants
YOUTUBE_API_BASE_URL = "https://www.googleapis.com/youtube/v3"
BATCH_SIZE = 50
LANGUAGE_CONFIDENCE_THRESHOLD = 0.95
ENGLISH_PATTERN = re.compile(r'^[A-Za-z0-9\s\-\.\,\!\?\:\;\'\"\/\(\)\#\@\&\*\+\=\%\$\[\]\{\}\|\\\~\`\^\<\>\_]+$')
TITLE_CLEANUP_PATTERN = re.compile(r'[^A-Za-z0-9\s\-\.\,\!\?\:\;\'\"\/\(\)\#\@\&\*\+\=\%\$\[\]\{\}\|\\\~\`\^\<\>\_]+')

# YouTube API constants
VALID_ORDERS = {"date", "rating", "relevance", "title", "videoCount", "viewCount"}
VIDEO_CATEGORIES = {
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
DURATION_MAP = ("short", "medium", "long")


class YouTubeStatsCollector:
    """
    Collects YouTube video statistics and metadata from channels.
    
    Supports API key rotation, quota management, and automatic translation
    of non-English video titles.
    """
    
    def __init__(self) -> None:
        """Initialize the collector with API key rotation."""
        self.rotator = APIKeyRotator()
        self.translator = Translator()
    
    # --------------------------------------------------------------------- #
    # API Communication
    # --------------------------------------------------------------------- #
    
    def _make_api_request(self, endpoint: str, params: dict, retries: int) -> dict:
        """
        Make a YouTube API request with automatic key rotation on 403 errors.
        
        Args:
            endpoint: API endpoint (e.g., 'search', 'videos')
            params: Request parameters
            retries: Number of retries remaining
            
        Returns:
            API response data
            
        Raises:
            QuotaExhaustedError: When all API keys have exhausted their quota
        """
        if retries == 0:
            raise QuotaExhaustedError("All API keys have exhausted their quota")
            
        params["key"] = self.rotator.current_key()
        url = f"{YOUTUBE_API_BASE_URL}/{endpoint}"
        
        response = requests.get(url, params=params)
        
        if response.status_code == 403:
            self.rotator.rotate_key()
            return self._make_api_request(endpoint, params, retries - 1)

        response.raise_for_status()
        return response.json()

    # --------------------------------------------------------------------- #
    # Channel Discovery
    # --------------------------------------------------------------------- #
    
    def fetch_channel_ids(
        self,
        keywords: str,
        video_duration: str,
        order: str,
        video_category_id: Optional[int],
        max_results_per_page: int = 50,
        max_channels: Optional[int] = None
    ) -> List[str]:
        """
        Fetch channel IDs from YouTube search results.
        
        Args:
            keywords: Search keywords/query
            video_duration: Duration filter (short/medium/long)
            order: Sort order for results
            video_category_id: Video category filter
            max_results_per_page: Results per API call
            max_channels: Maximum channels to return (None for all)
            
        Returns:
            List of channel IDs
        """
        params = {
            "part": "snippet",
            "order": order,
            "type": "video",
            "relevanceLanguage": "en",
            "regionCode": "US",
            "videoCategoryId": str(video_category_id) if video_category_id else "28",
            "videoDuration": video_duration,
            "q": keywords,
            "maxResults": str(max_results_per_page),
        }

        channels: List[str] = []
        next_page_token: Optional[str] = None

        while True:
            if next_page_token:
                params["pageToken"] = next_page_token
            else:
                params.pop("pageToken", None)

            data = self._make_api_request(
                "search",
                params=params,
                retries=len(self.rotator.keys)
            )

            # Extract channel IDs from this page
            for item in data.get("items", []):
                channel_id = item["snippet"]["channelId"]
                if channel_id not in channels:  # Avoid duplicates
                    channels.append(channel_id)
                    
                if max_channels and len(channels) >= max_channels:
                    return channels[:max_channels]

            # Check for next page
            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                break
                
        return channels

    def fetch_uploads_playlist(self, channel_id: str) -> str:
        """
        Get the uploads playlist ID for a channel.
        
        Args:
            channel_id: YouTube channel ID
            
        Returns:
            Uploads playlist ID
        """
        print(f"Fetching uploads playlist for {channel_id}...")
        
        data = self._make_api_request(
            "channels",
            {"part": "contentDetails", "id": channel_id},
            retries=len(self.rotator.keys)
        )
        
        return data["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    def fetch_channel_video_ids(self, uploads_playlist: str, max_videos: int) -> List[str]:
        """
        Get video IDs from a channel's uploads playlist.
        
        Args:
            uploads_playlist: Uploads playlist ID
            max_videos: Maximum number of videos to fetch
            
        Returns:
            List of video IDs
        """
        print(f"Fetching video IDs from uploads playlist {uploads_playlist}...")
        
        video_ids: List[str] = []
        next_token: Optional[str] = None

        while len(video_ids) < max_videos:
            params = {
                "part": "contentDetails",
                "playlistId": uploads_playlist,
                "maxResults": BATCH_SIZE,
            }
            
            if next_token:
                params["pageToken"] = next_token

            data = self._make_api_request(
                "playlistItems",
                params,
                retries=len(self.rotator.keys)
            )
            
            video_ids.extend(
                item["contentDetails"]["videoId"]
                for item in data.get("items", [])
            )
            
            next_token = data.get("nextPageToken")
            if not next_token:
                break

        return video_ids[:max_videos]

    # --------------------------------------------------------------------- #
    # Video Processing
    # --------------------------------------------------------------------- #
    
    def _detect_language(self, text: str) -> str:
        """
        Detect the language of text with confidence threshold.
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code (e.g., 'en', 'es', etc.)
        """
        try:
            languages = detect_langs(text)
            if not languages:
                return "en"
                
            top_language = languages[0]
            
            # If confidence is low, look for non-English alternatives
            if top_language.prob < LANGUAGE_CONFIDENCE_THRESHOLD:
                for lang_obj in languages:
                    if lang_obj.lang != 'en':
                        return lang_obj.lang
                return top_language.lang
            
            # High confidence case - validate English pattern
            is_english = (
                top_language.lang == 'en' and 
                bool(ENGLISH_PATTERN.fullmatch(text))
            )
            return "en" if is_english else top_language.lang
            
        except LangDetectException as e:
            print(f"Language detection error: {e}")
            return "en"

    async def _clean_title(self, title: str) -> str:
        """
        Clean and normalize video title.
        
        Args:
            title: Raw video title
            
        Returns:
            Cleaned title
        """
        # Remove emojis
        title = demoji.replace(title, "")
        
        # Translate if not English
        if self._detect_language(title) != "en":
            print(f"Translating title: {title}")
            translation = await self.translator.translate(title, dest='en')
            title = translation.text
        
        # Clean unwanted characters
        return TITLE_CLEANUP_PATTERN.sub('', title)

    async def fetch_video_pairs(self, video_ids: List[str]) -> Optional[List[List]]:
        """
        Fetch video titles and view counts.
        
        Args:
            video_ids: List of video IDs
            
        Returns:
            List of [title, view_count] pairs or None on error
        """
        try:
            pairs: List[List] = []
            print(f"Fetching video pairs for {len(video_ids)} videos...")

            # Process in batches
            for i in range(0, len(video_ids), BATCH_SIZE):
                chunk = video_ids[i:i + BATCH_SIZE]
                
                data = self._make_api_request(
                    "videos",
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
                    cleaned_title = await self._clean_title(raw_title)
                    views = int(item["statistics"].get("viewCount", 0))
                    
                    pairs.append([cleaned_title, views])
                    print(f"Processed video: {cleaned_title} ({views} views)")
                    
            return pairs
            
        except Exception as e:
            print(f"Error processing videos: {e}")
            return None

    # --------------------------------------------------------------------- #
    # Parameter Validation
    # --------------------------------------------------------------------- #
    
    def _validate_search_params(
        self, 
        keywords: str,
        video_duration: int,
        order: str,
        video_category_id: Optional[int]
    ) -> Tuple[str, str, Optional[int]]:
        """
        Validate and process search parameters.
        
        Args:
            keywords: Search keywords
            video_duration: Duration type (0=short, 1=medium, 2=long)
            order: Sort order
            video_category_id: Category ID
            
        Returns:
            Processed parameters
            
        Raises:
            ValueError: For invalid parameters
        """
        if order not in VALID_ORDERS:
            raise ValueError(f"order must be one of: {', '.join(sorted(VALID_ORDERS))}")
        
        if video_category_id is not None and video_category_id not in VIDEO_CATEGORIES:
            valid_ids = sorted(VIDEO_CATEGORIES.keys())
            raise ValueError(f"video_category_id must be one of: {valid_ids} or None")
        
        if video_duration not in [0, 1, 2]:
            raise ValueError("video_duration must be 0 (short), 1 (medium), or 2 (long)")
        
        processed_duration = DURATION_MAP[video_duration]
        return keywords, processed_duration, video_category_id

    # --------------------------------------------------------------------- #
    # Main Pipeline
    # --------------------------------------------------------------------- #
    
    async def build_results(
        self,
        num_channels: int,
        max_videos_per_channel: int,
        keywords: str = "technology",
        duration_type: int = 1,
        order: str = "relevance",
        video_category_id: Optional[int] = None
    ) -> Dict[str, List[List]]:
        """
        Execute the full data collection pipeline.
        
        Args:
            num_channels: Number of channels to process
            max_videos_per_channel: Max videos per channel
            keywords: Search keywords
            duration_type: Video duration filter
            order: Sort order
            video_category_id: Category filter
            
        Returns:
            Dictionary mapping channel IDs to video data
        """
        _, video_duration, video_category_id = self._validate_search_params(
            keywords, duration_type, order, video_category_id
        )
        
        results: Dict[str, List[List]] = {}
        
        try:
            channel_ids = self.fetch_channel_ids(
                keywords=keywords,
                video_duration=video_duration,
                order=order,
                video_category_id=video_category_id,
                max_channels=num_channels
            )
        except QuotaExhaustedError:
            print("API quota exhausted while fetching channel IDs. Saving partial results.")
            self.save_to_json(results)
            raise

        for channel_id in channel_ids:
            print(f"Processing channel {channel_id}...")
            try:
                uploads_playlist = self.fetch_uploads_playlist(channel_id)
                video_ids = self.fetch_channel_video_ids(uploads_playlist, max_videos_per_channel)
                video_pairs = await self.fetch_video_pairs(video_ids)
                
                if video_pairs:
                    results[channel_id] = video_pairs
                    
            except QuotaExhaustedError:
                print(f"API quota exhausted while processing channel {channel_id}. Saving partial results.")
                self.save_to_json(results)
                raise
                
        return results

    # --------------------------------------------------------------------- #
    # Data Persistence
    # --------------------------------------------------------------------- #
    
    def save_to_json(self, results: Dict, path: str = "data/results.json") -> None:
        """
        Save results to JSON file, merging with existing data.
        
        Args:
            results: Data to save
            path: Output file path (defaults to data/results.json)
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        existing_data = {}
        
        # Load existing data if file exists
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing_data = {}
        
        # Check for duplicates
        duplicate_keys = set(existing_data.keys()) & set(results.keys())
        if duplicate_keys:
            print(f"Warning: Overwriting existing keys: {duplicate_keys}")
        
        # Merge and save
        existing_data.update(results)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
            
        print(f"Saved {len(results)} channels to {path}")

    def get_subscriber_counts(self, channel_ids: List[str]) -> Dict[str, int]:
        """
        Fetch subscriber counts for channels.
        
        Args:
            channel_ids: List of channel IDs
            
        Returns:
            Dictionary mapping channel IDs to subscriber counts
        """
        data = self._make_api_request(
            "channels",
            {
                "part": "statistics",
                "id": ",".join(channel_ids)
            },
            retries=len(self.rotator.keys)
        )
        
        return {
            item["id"]: int(item["statistics"].get("subscriberCount", 0))
            for item in data.get("items", [])
        }