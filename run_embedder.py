#!/usr/bin/env python3
"""
Weaviate Embedder Runner

This script runs the Weaviate embedding process to load YouTube data
into the vector database for semantic search.
"""

from WeaviateEmbedder import WeaviateEmbedder

def main():
    """Main function to run the embedding process."""
    # Configuration
    FILE_PATH = "data/results.json"
    
    # Initialize embedder
    embedder = WeaviateEmbedder()
    
    try:
        # Embed data incrementally (won't delete existing data)
        total_videos_embedded = embedder.embed_data(FILE_PATH, force_recreate=True)
        
        # Get stats
        total_videos = embedder.get_stats()
        print(f"\n✅ Successfully embedded videos into Weaviate! Total videos: {total_videos}")
        print(f"✅ New videos embedded: {total_videos_embedded}")
        
        # Example search
        print(f"\n🔍 Example search for 'anime':")
        similar_videos = embedder.search_similar_videos("anime", limit=5)
        for i, video in enumerate(similar_videos, 1):
            # Access properties using .properties attribute
            title = video.properties['title']
            view_count = video.properties['viewCount']
            print(f"{i}. {title} ({view_count:,} views)")
            
            # Optional: also show the similarity distance
            if hasattr(video, 'metadata') and hasattr(video.metadata, 'distance'):
                print(f"   Distance: {video.metadata.distance:.4f}")
        
        # Uncomment the line below if you want to force recreate everything
        # embedder.embed_data("data/r.json", force_recreate=True)
        
    except Exception as e:
        print(f"Error in main process: {e}")
        raise
    finally:
        # Always close the connection
        embedder.close()


if __name__ == "__main__":
    main() 