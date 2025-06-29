"""
YouTube Stats Scraper Runner

This script runs the YouTube statistics collector with predefined parameters
and saves results to the data folder.
"""

import os
from datetime import datetime
from pathlib import Path

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


def main():
    """Main execution function."""
    print("Starting YouTube Stats Collection...")
    
    # Ensure data directory exists
    data_dir = ensure_data_directory()
    
    # Configuration
    config = {
        "num_channels": 950,
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
    filename = generate_filename(config["keywords"])
    output_path = data_dir / filename
    
    print(f"\nOutput will be saved to: {output_path}")
    
    try:
        # Initialize collector
        collector = YouTubeStatsCollector()
        
        # Run the collection
        print("\nStarting data collection...")
        results = collector.build_results(**config)
        
        # Save results
        print(f"\nSaving {len(results)} channels to {output_path}...")
        collector.save_to_json(results, str(output_path))
        
        print(f"\n✅ Collection completed successfully!")
        print(f"📊 Collected data from {len(results)} channels")
        print(f"💾 Results saved to: {output_path}")
        
        # Print summary statistics
        total_videos = sum(len(videos) for videos in results.values())
        print(f"📹 Total videos processed: {total_videos}")
        
        if results:
            avg_videos_per_channel = total_videos / len(results)
            print(f"📈 Average videos per channel: {avg_videos_per_channel:.1f}")
        
    except Exception as e:
        print(f"\n❌ Error during collection: {e}")
        print("Partial results may have been saved to the data folder.")
        raise


if __name__ == "__main__":
    main() 