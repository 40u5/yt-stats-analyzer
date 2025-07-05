import json
import weaviate
from sentence_transformers import SentenceTransformer
from typing import List
from weaviate.classes.config import Property, Configure
from weaviate.classes.config import DataType

MODEL_NAME = "all-mpnet-base-v2"


class WeaviateEmbedder:
    def __init__(self):
        """Initialize Weaviate client and sentence transformer model."""
        self.client = weaviate.connect_to_local(host="localhost", port=8080)
        self.model = SentenceTransformer(MODEL_NAME)
        print("Initialized Weaviate client and sentence transformer model")
    
    def create_schema(self, force_recreate: bool = False):
        """Create the schema for YouTube videos in Weaviate."""
        try:
            collection_name = "YouTubeVideo"
            
            # Check if collection already exists
            try:
                if force_recreate:
                    # Only delete if explicitly requested
                    self.client.collections.delete(collection_name)
                    print("Deleted existing YouTubeVideo class (force recreate)")
                else:
                    #logger.info("YouTubeVideo collection already exists, skipping schema creation")
                    return
            except:
                # Collection doesn't exist, proceed with creation
                pass
            
            # Create new class using v4 API
            self.client.collections.create(
                name=collection_name,
                description="YouTube video with title and view count",
                properties=[
                    Property(name="title", data_type=DataType.TEXT, description="The title of the YouTube video"),
                    Property(name="viewCount", data_type=DataType.INT, description="Number of views for the video"),
                    Property(name="channelId", data_type=DataType.TEXT, description="YouTube channel ID"),
                    Property(name="subscriberCount", data_type=DataType.INT, description="Number of subscribers for the channel"),
                    Property(name="videoId", data_type=DataType.TEXT, description="Unique video identifier"),
                ],
                vectorizer_config=Configure.Vectorizer.none()
            )
            print("Created YouTubeVideo schema successfully")
        except Exception as e:
            print(f"Error creating schema: {e}")
            raise
    
    def generate_embeddings(self, titles: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of titles."""
        print(f"Generating embeddings for {len(titles)} titles")
        embeddings = self.model.encode(titles, convert_to_tensor=False)
        return embeddings.tolist()
    
    def get_existing_video_ids(self) -> set:
        """Get all existing video IDs to avoid duplicates."""
        try:
            collection = self.client.collections.get("YouTubeVideo")
        except Exception as e:
            print(f"Could not access YouTubeVideo collection: {e}")
            return set()

        existing_ids = set()
        try:
            # Use the v4 iterator to page through *all* objects without offset limits
            for obj in collection.iterator(
                    include_vector=False,           # skip vectors for speed
                    return_properties=["videoId"],  # only fetch what you need
                    cache_size=1000                 # how many objects per RPC
                ):
                vid = obj.properties.get("videoId")
                if vid:
                    existing_ids.add(vid)
        except Exception as e:
            print(f"Error while fetching video IDs: {e}")
            # Depending on your needs, you could choose to re-raise, return partial results, or return empty
            # Here we return whatever we've collected so far:
        
        print(f"Found {len(existing_ids)} existing videos to check against for duplicates")
        return existing_ids
    
    def embed_data(self, data_file: str, force_recreate: bool = False):
        """Load data and embed it into Weaviate with duplicate detection."""
        # Create schema if needed
        self.create_schema(force_recreate=force_recreate)
        
        # Load the JSON data
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded data for {len(data)} channels")
        
        # Get existing video IDs to avoid duplicates
        existing_video_ids = self.get_existing_video_ids()
        
        # Prepare data for embedding
        videos_to_embed = []
        
        for channel_id, channel_data in data.items():
            subscriber_count = channel_data.get('subscriber_count', 0)
            
            for video in channel_data.get('videos', []):
                if len(video) >= 2:
                    title, view_count = video[0], video[1]
                    
                    video_id = f"{channel_id}_{title}"
                    
                    # Skip if this video already exists
                    if video_id in existing_video_ids:
                        print(f"Skipping duplicate video: {title}")
                        continue
                    
                    videos_to_embed.append({
                        'title': title,
                        'viewCount': view_count,
                        'channelId': channel_id,
                        'subscriberCount': subscriber_count,
                        'videoId': video_id
                    })
        
        if not videos_to_embed:
            print("No new videos to embed (all are duplicates)")
            return 0
        
        print(f"Prepared {len(videos_to_embed)} new videos for embedding")
        
        # Generate embeddings for all titles
        titles = [video['title'] for video in videos_to_embed]
        embeddings = self.generate_embeddings(titles)
        
        # Batch insert into Weaviate
        batch_size = 100
        collection = self.client.collections.get("YouTubeVideo")
        
        for i in range(0, len(videos_to_embed), batch_size):
            batch = videos_to_embed[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            
            with collection.batch.dynamic() as batch_client:
                for video, embedding in zip(batch, batch_embeddings):
                    batch_client.add_object(
                        properties=video,
                        vector=embedding
                    )
            
            print(f"Inserted batch {i//batch_size + 1}/{(len(videos_to_embed) + batch_size - 1)//batch_size}")
        
        print(f"Successfully embedded {len(videos_to_embed)} new videos into Weaviate")
        return len(videos_to_embed)
    
    def search_similar_videos(self, query: str, limit: int = 5):
        """Search for videos similar to the given query."""
        # Generate embedding for the query
        query_embedding = self.model.encode([query], convert_to_tensor=False).tolist()[0]
        
        # Search in Weaviate using the v4 query API
        collection = self.client.collections.get("YouTubeVideo")
        result = collection.query.near_vector(
            near_vector=query_embedding,  # Use 'near_vector' parameter name
            limit=limit,
            return_metadata=['distance']  # Use return_metadata instead of with_additional
        )
        return result.objects

    def get_stats(self):
        """Get statistics about the embedded data."""
        try:
            collection = self.client.collections.get("YouTubeVideo")
            result = collection.aggregate.over_all(total_count=True)
            total_count = result.total_count
            print(f"Total videos in database: {total_count}")
            return total_count
        except Exception as e:
            print(f"Error getting stats: {e}")
            return 0
    
    def close(self):
        """Close the Weaviate client connection."""
        try:
            if hasattr(self, 'client') and self.client:
                self.client.close()
                print("Weaviate client connection closed")
        except Exception as e:
            print(f"Error closing Weaviate client: {e}")

    def embed_videos_for_channel(self, channel_id: str, subscriber_count: int, video_pairs: list):
        """Embed a list of videos for a channel directly into Weaviate."""
        self.create_schema(force_recreate=False)
        existing_video_ids = self.get_existing_video_ids()
        videos_to_embed = []
        skipped_count = 0
        
        for video in video_pairs:
            if len(video) >= 2:
                title, view_count = video[0], video[1]
                video_id = f"{channel_id}_{title}"
                if video_id in existing_video_ids:
                    skipped_count += 1
                    continue
                videos_to_embed.append({
                    'title': title,
                    'viewCount': view_count,
                    'channelId': channel_id,
                    'subscriberCount': subscriber_count,
                    'videoId': video_id
                })
        
        if not videos_to_embed:
            print(f"📋 Channel {channel_id}: {len(video_pairs)} videos fetched, {skipped_count} duplicates skipped, 0 new videos embedded")
            return 0
            
        titles = [video['title'] for video in videos_to_embed]
        embeddings = self.generate_embeddings(titles)
        batch_size = 100
        collection = self.client.collections.get("YouTubeVideo")
        
        for i in range(0, len(videos_to_embed), batch_size):
            batch = videos_to_embed[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            with collection.batch.dynamic() as batch_client:
                for video, embedding in zip(batch, batch_embeddings):
                    batch_client.add_object(
                        properties=video,
                        vector=embedding
                    )
        
        print(f"📋 Channel {channel_id}: {len(video_pairs)} videos fetched, {skipped_count} duplicates skipped, {len(videos_to_embed)} new videos embedded")
        
        # Show the titles of videos that were embedded
        if videos_to_embed:
            print("  New videos embedded:")
            for video in videos_to_embed:
                print(f"{video['title']} ({video['viewCount']:,} views)")
        
        return len(videos_to_embed) 