#!/usr/bin/env python3
"""
Weaviate Embedder Runner

This script runs the Weaviate embedding process to load YouTube data
into the vector database for semantic search.
"""

from WeaviateEmbedder import WeaviateEmbedder

def main():
    """Main function to run the embedding process."""
    
    # Initialize embedder
    embedder = WeaviateEmbedder()
    
    try:

        # Get stats
        total_videos = embedder.get_stats()
        print(f"\n✅ Successfully embedded videos into Weaviate! Total videos: {total_videos}")
        
        # Example search
        print(f"\n🔍 Example search for 'SteamOS':")
        similar_videos = embedder.search_similar_videos("girlfriend", limit=5)
        for i, video in enumerate(similar_videos, 1):
            # Access properties using .properties attribute
            title = video.properties['title']
            view_count = video.properties['viewCount']
            print(f"{i}. {title} ({view_count:,} views)")
            
            # Optional: also show the similarity distance
            if hasattr(video, 'metadata') and hasattr(video.metadata, 'distance'):
                print(f"   Distance: {video.metadata.distance:.4f}")        
    except Exception as e:
        print(f"Error in main process: {e}")
        raise
    finally:
        # Always close the connection
        embedder.close()


if __name__ == "__main__":
    main() 