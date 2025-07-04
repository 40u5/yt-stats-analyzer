# YouTube Video Embedding with Weaviate

This project embeds YouTube video titles into a Weaviate vector database for semantic search and similarity analysis. It includes a comprehensive YouTube data collection system with API key rotation, automatic translation, and direct embedding capabilities.

## Prerequisites

1. **Weaviate Docker Container**: Make sure your Weaviate container is running
2. **Python Dependencies**: All required packages are in `requirements.txt`
3. **YouTube API Keys**: You'll need one or more YouTube Data API v3 keys

## Setup

1. **Start Weaviate** (if not already running):
```bash
docker run -d \
  --name weaviate \
  -p 8080:8080 \
  -e QUERY_DEFAULTS_LIMIT=25 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \
  -e DEFAULT_VECTORIZER_MODULE='none' \
  -e ENABLE_MODULES='' \
  -e CLUSTER_HOSTNAME='node1' \
  semitechnologies/weaviate:1.22.4
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure API Keys**:
   - The system will automatically prompt you to create a `.env` file on first run
   - Enter your YouTube API keys when prompted (comma-separated for multiple keys)
   - The API key rotator will manage quota across multiple keys automatically

## Usage

### 1. Collect YouTube Data and Embed Directly

Run the main scraper to collect YouTube data and embed it directly into Weaviate:

```bash
python run_scraper.py
```

This script will:
- Search for YouTube channels based on your keywords
- Collect video data from each channel with subscriber counts
- Embed videos directly into Weaviate using the `all-mpnet-base-v2` model
- Show progress and comprehensive statistics
- Handle API quota management with automatic key rotation

#### Infinite Mode

You can enable infinite mode to collect all available channels until API quota is exhausted:

```python
# In run_scraper.py, set:
config = {
    "num_channels": True,  # Infinite mode - collects until quota exhausted
    "max_videos_per_channel": 10,
    "keywords": "AI",
    # ... other settings
}
```

**Infinite Mode Features:**
- ♾️ Collects channels continuously until API quota is exhausted
- 🔄 Uses multiple search strategies to maximize channel discovery
- 📊 Provides detailed progress updates
- 🛡️ Handles errors gracefully and continues with next channel
- 📈 Shows comprehensive statistics at completion
- 🌐 Automatic translation of non-English video titles
- 🔑 Automatic API key rotation when quota is exhausted

**Finite Mode:**
```python
config = {
    "num_channels": 5,  # Collect exactly 5 channels
    # ... other settings
}
```

### 2. Embed Existing Data into Weaviate (Legacy)

Run the legacy embedding script to load existing JSON data into Weaviate:

```bash
python run_embedder.py
```

This script will:
- Connect to your Weaviate instance (default: `http://localhost:8080`)
- Create a `YouTubeVideo` schema with properties for title, view count, channel ID, subscriber count, and video ID
- Generate embeddings for all video titles using the `all-mpnet-base-v2` model
- Batch insert all videos into the vector database with duplicate detection
- Show statistics and perform an example search

## Data Schema

The `YouTubeVideo` class in Weaviate contains:

- **title** (text): Video title (automatically translated to English if needed)
- **viewCount** (int): Number of views
- **channelId** (text): YouTube channel ID
- **subscriberCount** (int): Channel subscriber count
- **videoId** (text): Unique video identifier
- **vector** (768-dimensional): Title embedding using `all-mpnet-base-v2`

## Features

### YouTube Stats Collector (`YoutubeStatsCollector.py`)
- ✅ API key rotation with quota management
- ✅ Automatic translation of non-English video titles
- ✅ Multiple search strategies for comprehensive channel discovery
- ✅ Subscriber count collection
- ✅ Direct Weaviate embedding integration
- ✅ Error handling and graceful degradation
- ✅ Progress logging and statistics

### Embedding Script (`WeaviateEmbedder.py` and `run_embedder.py`)
- ✅ Automatic schema creation
- ✅ Batch processing for efficiency
- ✅ Duplicate detection to avoid re-embedding
- ✅ Progress logging
- ✅ Error handling
- ✅ Example search demonstration
- ✅ Statistics and monitoring

### API Key Rotator (`api_key_rotator.py`)
- ✅ Secure API key management
- ✅ Automatic key rotation on quota exhaustion
- ✅ Usage tracking and monitoring
- ✅ Environment-based configuration
- ✅ Input validation and error handling

## Configuration

### Scraper Configuration

In `run_scraper.py`, you can configure the collection parameters:

```python
config = {
    "num_channels": True,  # True for infinite mode, number for finite mode
    "max_videos_per_channel": 10,  # Videos per channel to collect
    "keywords": "AI",  # Search keywords
    "duration_type": 1,  # 0=short, 1=medium, 2=long
    "order": "relevance",  # relevance, date, viewCount, rating, title
    "video_category_id": 28,  # YouTube category ID (28=Science & Technology)
    "write_to_weaviate": True  # Embed directly to Weaviate
}
```

### Weaviate Configuration

You can modify the Weaviate connection URL in both scripts:

```python
# Default: http://localhost:8080
embedder = WeaviateEmbedder("http://your-weaviate-url:8080")
```

### API Key Configuration

The system automatically manages API keys through the `.env` file:

```bash
# .env file format
API_KEYS=key1,key2,key3
```

## Example Queries

You can query the embedded data programmatically:

```python
from WeaviateEmbedder import WeaviateEmbedder

embedder = WeaviateEmbedder()

# Find videos similar to "AI and machine learning"
videos = embedder.search_similar_videos("AI and machine learning", limit=5)

# Get statistics
total_videos = embedder.get_stats()
```

## Troubleshooting

1. **Connection Error**: Make sure Weaviate is running on the correct port
2. **Schema Error**: The script will automatically recreate the schema if needed
3. **Memory Issues**: Reduce batch size in `WeaviateEmbedder.py` if processing large datasets
4. **API Quota Exhausted**: The system will automatically rotate keys, but ensure you have multiple valid API keys
5. **Translation Errors**: Non-English titles are automatically translated, but network issues may cause failures

## Performance Notes

- The `all-mpnet-base-v2` model provides excellent semantic understanding with 768-dimensional vectors
- Batch processing is used to optimize insertion performance
- Vector similarity search is very fast for semantic queries
- API key rotation ensures maximum data collection before quota exhaustion
- Duplicate detection prevents re-processing of existing videos

## Data Collection Statistics

The system provides comprehensive statistics including:
- Total channels processed
- Total videos embedded
- Average videos per channel
- Average subscribers per channel
- Top channels by subscriber count
- API key usage statistics

## Next Steps

You can extend this system by:
- Adding more metadata fields (upload date, duration, likes, comments, etc.)
- Implementing hybrid search (vector + keyword)
- Creating a web interface for searching
- Adding video content analysis
- Implementing recommendation systems
- Adding sentiment analysis of video titles
- Creating dashboards for YouTube analytics 