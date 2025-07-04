"""
YouTube Stats Scraper Runner

This script runs the YouTube statistics collector with predefined parameters
and saves results to the data folder, including subscriber counts.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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


async def main():
    """Main execution function."""
    print("Starting YouTube Stats Collection...")
    
    # Ensure data directory exists
    data_dir = ensure_data_directory()
    write_to_weaviate = True
    # Configuration
    # Set num_channels to True for infinite mode (keeps collecting until API quota is exhausted)
    # Set num_channels to a number (e.g., 5, 10, 100) for finite mode
    config = {
        "num_channels": 2,  # Set to True for infinite mode, or a number for finite mode
        "max_videos_per_channel": 10,
        "keywords": "AI",
        "duration_type": 1,  # 0=short, 1=medium, 2=long
        "order": "relevance",
        "video_category_id": 28,  # Science & Technology
        "write_to_weaviate": write_to_weaviate
    }
    
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    collector = None
    try:
        # Initialize collector
        collector = YouTubeStatsCollector()
        
        # Run the collection
        print("\nStarting data collection...")
        results = await collector.build_results(**config)
        
        # Embed videos directly into Weaviate
        print(f"\nEmbedding videos directly into Weaviate...")
        
        
        if not write_to_weaviate and results:
            # Print summary statistics
            total_subscribers = sum(results[ch]["subscriber_count"] for ch in results)
            total_videos = sum(len(results[ch]["videos"]) for ch in results)
            print(f"📹 Total videos processed: {total_videos}")
            avg_videos_per_channel = total_videos / len(results)
            avg_subscribers_per_channel = total_subscribers / len(results)
            print(f"📈 Average videos per channel: {avg_videos_per_channel:.1f}")
            print(f"👤 Average subscribers per channel: {avg_subscribers_per_channel:,.0f}")
            # Show top channels by subscriber count
            sorted_channels = sorted(
            results.items(), 
            key=lambda x: x[1]["subscriber_count"], 
            reverse=True
            )
        
            print(f"\n🏆 Top 5 channels by subscriber count:")
            for i, (channel_id, channel_data) in enumerate(sorted_channels[:5], 1):
                sub_count = channel_data["subscriber_count"]
                video_count = len(channel_data["videos"])
                print(f"  {i}. {channel_id}: {sub_count:,} subscribers, {video_count} videos")

        elif write_to_weaviate:
            print(f"\n🎉 Collection Summary:")
            print(f"📺 Channels embedded: {results['total_channels']}")
            print(f"📹 Videos embedded: {results['total_videos']}")
            if config["num_channels"] is True:
                print(f"♾️  Infinite mode completed - all available channels processed")
            else:
                print(f"✅ Finite mode completed - {config['num_channels']} channels requested")
        
    except Exception as e:
        print(f"\n❌ Error during collection: {e}")
        print("Partial results may was saved to the data folder.")
        raise
    finally:
        # Always close the collector to clean up resources
        if collector:
            await collector.aclose()
            print("🧹 Resources cleaned up successfully")


if __name__ == "__main__":
    # Set up proper asyncio event loop cleanup
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}") 