"""
YouTube Statistics Collector

A tool for collecting YouTube video statistics and metadata from channels
based on search criteria. Supports API key rotation and quota management.
"""

import json
import os
import re
from typing import Optional, Dict, List, Tuple, Union

import requests
from requests.adapters import HTTPAdapter

from api_key_rotator import APIKeyRotator
from exceptions import QuotaExhaustedError
from WeaviateEmbedder import WeaviateEmbedder
from TranslationHandler import TranslationHandler


# Constants
YOUTUBE_API_BASE_URL = "https://www.googleapis.com/youtube/v3"
BATCH_SIZE = 50

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
        self.translation_handler = TranslationHandler()
        self.embedder = WeaviateEmbedder()
        # Create a session for better connection management
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'YouTube-Stats-Collector/1.0'
        })
        # Configure session for better connection handling
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
    
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
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 403:
                self.rotator.rotate_key()
                return self._make_api_request(endpoint, params, retries - 1)

            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            raise

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
        page_token: Optional[str] = None
    ) -> Tuple[List[str], Optional[str]]:
        """
        Fetch a single page of channel IDs from YouTube search results.
        Args:
            keywords: Search keywords/query
            video_duration: Duration filter (short/medium/long)
            order: Sort order for results
            video_category_id: Video category filter
            max_results_per_page: Results per API call
            page_token: Token for the next page (None for first page)
        Returns:
            Tuple of (list of channel IDs, nextPageToken or None)
        """
        params = {
            "part": "snippet",
            "type": "video",
            "relevanceLanguage": "en",
            "regionCode": "US",
            "videoCategoryId": str(video_category_id) if video_category_id else "28",
            "q": keywords,
            "maxResults": str(max_results_per_page),
            "order": order,
            "videoDuration": video_duration,
        }
        if page_token:
            params["pageToken"] = page_token
        print(f"🔍 Fetching channel page (order={order}, duration={video_duration}, page_token={page_token})")
        data = self._make_api_request(
            "search",
            params=params,
            retries=len(self.rotator.keys)
        )
        channel_ids = []
        for item in data.get("items", []):
            channel_id = item["snippet"]["channelId"]
            if channel_id not in channel_ids:
                channel_ids.append(channel_id)
        next_page_token = data.get("nextPageToken")
        print(f"  Found {len(channel_ids)} channels on this page. Next page token: {next_page_token}")
        return channel_ids, next_page_token

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
            print(f"Fetching video data for {len(video_ids)} videos...")

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
                    cleaned_title = await self.translation_handler.clean_title(raw_title)
                    views = int(item["statistics"].get("viewCount", 0))
                    
                    pairs.append([cleaned_title, views])
                    
            print(f"Fetched data for {len(pairs)} videos")
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
        num_channels: Union[int, bool],
        max_videos_per_channel: int,
        keywords: str = "technology",
        duration_type: int = 1,
        order: str = "relevance",
        video_category_id: Optional[int] = None, 
        write_to_weaviate: bool = False
    ) -> dict:
        """
        Execute the full data collection pipeline, embed directly to Weaviate, and return results.
        In infinite mode, fetch and process one page at a time, writing videos to weaviate, and continue with the next page until done.
        """
        _, video_duration, video_category_id = self._validate_search_params(
            keywords, duration_type, order, video_category_id
        )
        infinite_mode = num_channels is True
        max_channels = None if infinite_mode else num_channels
        results: Dict[str, Union[int, List[List]]] = {}
        total_channels = 0
        total_videos = 0
        if infinite_mode:
            print("🔄 Infinite mode enabled - collecting all available channels (one page at a time)...")
            page_token = None
            while True:
                try:
                    channel_ids, next_page_token = self.fetch_channel_ids(
                        keywords=keywords,
                        video_duration=video_duration,
                        order=order,
                        video_category_id=video_category_id,
                        page_token=page_token
                    )
                except QuotaExhaustedError:
                    print("API quota exhausted while fetching channel IDs. Stopping.")
                    break
                if not channel_ids:
                    print("No more channels found on this page. Stopping.")
                    break
                for channel_id in channel_ids:
                    print(f"\n📺 Processing channel: {channel_id}")
                    try:
                        uploads_playlist = self.fetch_uploads_playlist(channel_id)
                        video_ids = self.fetch_channel_video_ids(uploads_playlist, max_videos_per_channel)
                        video_pairs = await self.fetch_video_pairs(video_ids)
                        if video_pairs:
                            total_channels += 1
                            if write_to_weaviate:
                                subscriber_count = self.get_subscriber_counts([channel_id]).get(channel_id, 0)
                                videos_embedded = self.embedder.embed_videos_for_channel(channel_id, subscriber_count, video_pairs)
                                total_videos += videos_embedded
                            else:
                                results[channel_id] = video_pairs
                                total_videos += len(video_ids)
                            if write_to_weaviate:
                                print(f"📊 Progress: {total_channels} channels, {total_videos} new videos embedded")
                            else:
                                print(f"📊 Progress: {total_channels} channels, {total_videos} videos processed")
                    except QuotaExhaustedError:
                        print(f"❌ API quota exhausted while processing channel {channel_id}. Stopping.")
                        print(f"📈 Final stats: {total_channels} channels, {total_videos} videos processed")
                        return results
                    except Exception as e:
                        print(f"⚠️  Error processing channel {channel_id}: {e}")
                        print("Continuing with next channel...")
                        continue
                if not next_page_token:
                    print(f"\n✅ All pages processed!")
                    break
                page_token = next_page_token
            results['total_channels'] = total_channels
            results['total_videos'] = total_videos
            print(f"\n✅ Infinite collection completed!")
            if write_to_weaviate:
                print(f"📊 Final stats: {total_channels} channels, {total_videos} new videos embedded into Weaviate")
            else:
                print(f"📊 Final stats: {total_channels} channels, {total_videos} videos processed")
        else:
            try:
                # Ensure max_results_per_page is a valid int (YouTube API max is 50)
                max_results_per_page = 50 if not isinstance(max_channels, int) or max_channels is None or max_channels > 50 else max_channels
                channel_ids, _ = self.fetch_channel_ids(
                    keywords=keywords,
                    video_duration=video_duration,
                    order=order,
                    video_category_id=video_category_id,
                    max_results_per_page=max_results_per_page
                )
            except QuotaExhaustedError:
                print("API quota exhausted while fetching channel IDs. Stopping.")
                raise
            for i, channel_id in enumerate(channel_ids, 1):
                print(f"\n📺 Processing channel {i}/{len(channel_ids)}: {channel_id}")
                try:
                    uploads_playlist = self.fetch_uploads_playlist(channel_id)
                    video_ids = self.fetch_channel_video_ids(uploads_playlist, max_videos_per_channel)
                    video_pairs = await self.fetch_video_pairs(video_ids)
                    if video_pairs:
                        total_channels += 1
                        if write_to_weaviate:
                            subscriber_count = self.get_subscriber_counts([channel_id]).get(channel_id, 0)
                            videos_embedded = self.embedder.embed_videos_for_channel(channel_id, subscriber_count, video_pairs)
                            total_videos += videos_embedded
                        else:
                            results[channel_id] = video_pairs
                            total_videos += len(video_ids)
                        if write_to_weaviate:
                            print(f"📊 Progress: {total_channels} channels, {total_videos} new videos embedded")
                        else:
                            print(f"📊 Progress: {total_channels} channels, {total_videos} videos processed")
                except QuotaExhaustedError:
                    print(f"❌ API quota exhausted while processing channel {channel_id}. Stopping.")
                    print(f"📈 Final stats: {total_channels} channels, {total_videos} videos processed")
                    raise
                except Exception as e:
                    print(f"⚠️  Error processing channel {channel_id}: {e}")
                    print("Continuing with next channel...")
                    continue
            results['total_channels'] = total_channels
            results['total_videos'] = total_videos
            print(f"\n✅ Collection completed!")
            if write_to_weaviate:
                print(f"📊 Final stats: {total_channels} channels, {total_videos} new videos embedded into Weaviate")
            else:
                print(f"📊 Final stats: {total_channels} channels, {total_videos} videos processed")
        return results

    # --------------------------------------------------------------------- #
    # Data Persistence
    # --------------------------------------------------------------------- #
    
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
    
    async def aclose(self):
        """Properly close all resources including async clients."""
        print("🧹 Starting resource cleanup...")
        
        # 1) shut down your requests.Session
        if hasattr(self, 'session'):
            self.session.close()
            print("  ✅ Requests session closed")

        # 2) close the translation handler
        if hasattr(self, 'translation_handler'):
            try:
                await self.translation_handler.aclose()
            except Exception as e:
                print(f"  ⚠️  Warning: Error closing translation handler: {e}")

        # 3) for the embedder, just call its existing sync close()
        if hasattr(self.embedder, 'close'):
            try:
                self.embedder.close()
                print("  ✅ Weaviate embedder closed")
            except Exception as e:
                print(f"  ⚠️  Warning: Error closing Weaviate embedder: {e}")
        
        print("🧹 Resource cleanup completed")
