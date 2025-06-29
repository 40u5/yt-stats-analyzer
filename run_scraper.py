"""
YouTube Stats Scraper Runner

This script runs the YouTube statistics collector with predefined parameters
and saves results to the data folder, including subscriber counts.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import json
import os

from YoutubeStatsCollector import YouTubeStatsCollector


def ensure_data_directory():
    """Ensure the data directory exists."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    return data_dir


def generate_filename(keywords: str, timestamp: bool = True) -> str:
    """
    Generate a filename for the results.
    
    Args:
        keywords: Search keywords used
        timestamp: Whether to include timestamp in filename
        
    Returns:
        Generated filename
    """
    # Clean keywords for filename
    clean_keywords = keywords.replace(" ", "_").replace("&", "and").lower()
    
    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"youtube_stats_{clean_keywords}_{timestamp_str}.json"
    else:
        return f"youtube_stats_{clean_keywords}.json"


def fetch_subscriber_counts(collector: YouTubeStatsCollector, channel_ids: List[str]) -> Dict[str, int]:
    """
    Fetch subscriber counts for the given channel IDs.
    
    Args:
        collector: YouTubeStatsCollector instance
        channel_ids: List of channel IDs to fetch subscriber counts for
        
    Returns:
        Dictionary mapping channel IDs to subscriber counts
    """
    print(f"Fetching subscriber counts for {len(channel_ids)} channels...")
    
    # Process in batches to avoid API limits
    batch_size = 50  # YouTube API allows up to 50 channel IDs per request
    all_subscriber_counts = {}
    
    for i in range(0, len(channel_ids), batch_size):
        batch = channel_ids[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(channel_ids) + batch_size - 1)//batch_size}")
        
        try:
            batch_counts = collector.get_subscriber_counts(batch)
            all_subscriber_counts.update(batch_counts)
            print(f"  Successfully fetched {len(batch_counts)} subscriber counts")
        except Exception as e:
            print(f"  Error fetching batch: {e}")
            # Continue with other batches
    
    return all_subscriber_counts


def enhance_data_with_subscribers(
    data: Dict[str, List[List]], 
    subscriber_counts: Dict[str, int]
) -> Dict[str, Dict[str, Any]]:
    """
    Enhance the data structure to include subscriber counts.
    
    Args:
        data: Original data with channel IDs as keys and video lists as values
        subscriber_counts: Dictionary mapping channel IDs to subscriber counts
        
    Returns:
        Enhanced data structure with subscriber counts and video data
    """
    enhanced_data = {}
    
    for channel_id, videos in data.items():
        subscriber_count = subscriber_counts.get(channel_id, 0)
        
        enhanced_data[channel_id] = {
            "subscriber_count": subscriber_count,
            "videos": videos
        }
        
        print(f"Channel {channel_id}: {subscriber_count:,} subscribers, {len(videos)} videos")
    
    return enhanced_data


def save_enhanced_results(data: Dict[str, Any], output_path: str) -> None:
    """
    Save the enhanced results to JSON file, merging with existing data.
    
    Args:
        data: Enhanced data to save
        output_path: Output file path
    """
    existing_data = {}
    
    # Load existing data if file exists
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            print(f"Loaded existing data from {output_path}")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load existing data: {e}")
            existing_data = {}
    
    # Check for duplicates
    duplicate_keys = set(existing_data.keys()) & set(data.keys())
    if duplicate_keys:
        print(f"Warning: Overwriting existing channel data: {duplicate_keys}")
    
    # Merge and save
    existing_data.update(data)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)
    
    print(f"Enhanced data saved to {output_path} (merged with existing data)")


async def main():
    """Main execution function."""
    print("Starting YouTube Stats Collection...")
    
    # Ensure data directory exists
    data_dir = ensure_data_directory()
    
    # Configuration
    config = {
        "num_channels": 2,
        "max_videos_per_channel": 10,
        "keywords": "AI",
        "duration_type": 1,  # 0=short, 1=medium, 2=long
        "order": "relevance",
        "video_category_id": 28  # Science & Technology
    }
    
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Generate output filename
    filename = "r.json"
    output_path = data_dir / filename
    
    print(f"\nOutput will be saved to: {output_path}")
    
    try:
        # Initialize collector
        collector = YouTubeStatsCollector()
        
        # Run the collection
        print("\nStarting data collection...")
        results = await collector.build_results(**config)
        
        # Fetch subscriber counts for all channels
        channel_ids = list(results.keys())
        subscriber_counts = fetch_subscriber_counts(collector, channel_ids)
        
        # Enhance data structure with subscriber counts
        print("\nEnhancing data structure with subscriber counts...")
        enhanced_results = enhance_data_with_subscribers(results, subscriber_counts)
        
        # Save enhanced results
        print(f"\nSaving {len(enhanced_results)} channels to {output_path}...")
        save_enhanced_results(enhanced_results, str(output_path))
        
        print(f"\n✅ Collection completed successfully!")
        print(f"📊 Collected data from {len(enhanced_results)} channels")
        print(f"💾 Results saved to: {output_path}")
        
        # Print summary statistics
        total_subscribers = sum(enhanced_results[ch]["subscriber_count"] for ch in enhanced_results)
        total_videos = sum(len(enhanced_results[ch]["videos"]) for ch in enhanced_results)
        
        print(f"📹 Total videos processed: {total_videos}")
        
        if enhanced_results:
            avg_videos_per_channel = total_videos / len(enhanced_results)
            avg_subscribers_per_channel = total_subscribers / len(enhanced_results)
            print(f"📈 Average videos per channel: {avg_videos_per_channel:.1f}")
            print(f"👤 Average subscribers per channel: {avg_subscribers_per_channel:,.0f}")
        
        # Show top channels by subscriber count
        sorted_channels = sorted(
            enhanced_results.items(), 
            key=lambda x: x[1]["subscriber_count"], 
            reverse=True
        )
        
        print(f"\n🏆 Top 5 channels by subscriber count:")
        for i, (channel_id, channel_data) in enumerate(sorted_channels[:5], 1):
            sub_count = channel_data["subscriber_count"]
            video_count = len(channel_data["videos"])
            print(f"  {i}. {channel_id}: {sub_count:,} subscribers, {video_count} videos")
        
    except Exception as e:
        print(f"\n❌ Error during collection: {e}")
        print("Partial results may was saved to the data folder.")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 