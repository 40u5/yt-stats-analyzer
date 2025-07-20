# YouTube Video Title Distance Analyzer

A powerful tool for analyzing the semantic relationships between YouTube video titles using vector embeddings and distance metrics.

## Overview

This project analyzes the average distance between video titles to discover patterns and relations in video content. By converting video titles into high-dimensional vector embeddings, the tool can:

- **Measure Content Similarity**: Calculate cosine distances between video titles to understand how similar or diverse a channel's content is
- **Identify Content Patterns**: Find channels with highly similar vs. diverse video titles
- **Analyze Channel Strategies**: Understand if channels focus on niche topics or cover broad subjects
- **Discover Content Clusters**: Group videos by semantic similarity within channels

## Key Features

### 🔍 **Semantic Analysis**
- Uses state-of-the-art sentence transformers (`all-mpnet-base-v2`) to convert video titles into meaningful vector representations
- Calculates cosine distances between all video pairs within each channel
- Provides average distance metrics to quantify content diversity

### 📊 **Comprehensive Analytics**
- **Channel-level Analysis**: Compare content diversity across different YouTube channels
- **Subscriber Correlation**: Analyze relationship between subscriber count and content similarity
- **Statistical Insights**: Mean, median, standard deviation of distance metrics
- **Top/Bottom Performers**: Identify channels with highest and lowest content diversity

### 📈 **Visualization**
- Interactive scatter plots showing subscriber count vs. average video distance
- Logarithmic scaling for better visualization of large subscriber ranges
- Trend line analysis to identify patterns
- Statistical annotations on graphs

### 🗄️ **Data Management**
- Weaviate vector database integration for efficient storage and retrieval
- Duplicate detection to prevent redundant analysis
- Batch processing for large datasets
- Connection testing and error handling

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ YouTube Data    │───▶│ Weaviate Vector  │───▶│ Distance        │
│ Collection      │    │ Database         │    │ Analysis        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Title Embedding │    │ Vector Storage   │    │ Statistical     │
│ Generation      │    │ & Retrieval      │    │ Visualization   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd yt-virality-score
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Weaviate**
   - Install and start Weaviate (see [Weaviate documentation](https://weaviate.io/developers/weaviate/installation))
   - Ensure it's running on `localhost:8080`

## Usage

### Basic Analysis

Run the main analysis script to analyze all channels in your database:

```bash
python analyze_video_distances.py
```

This will:
- Connect to your Weaviate database
- Calculate average distances between video titles for each channel
- Generate visualizations and statistics
- Save results to `data/video_distance_analysis.png`

### Programmatic Usage

```python
from analyze_video_distances import VideoDistanceAnalyzer

# Initialize analyzer
analyzer = VideoDistanceAnalyzer()

# Analyze all channels
results = analyzer.analyze_all_channels()

# Create visualization
analyzer.create_visualization(results)

# Print summary statistics
analyzer.print_summary_statistics(results)

# Clean up
analyzer.close()
```

## Understanding the Results

### Distance Metrics
- **Lower Distance (0.0-0.3)**: Very similar content, niche channels
- **Medium Distance (0.3-0.7)**: Balanced content diversity
- **Higher Distance (0.7-1.0)**: Very diverse content, broad topics

### Key Insights
- **Content Strategy**: Channels with low distances often focus on specific niches
- **Audience Retention**: High diversity might indicate broader audience appeal
- **Content Planning**: Patterns can reveal systematic content strategies

## Data Requirements

Your Weaviate database should contain a `YouTubeVideo` collection with:
- `title`: Video title (text)
- `channelId`: YouTube channel identifier (text)
- `subscriberCount`: Channel subscriber count (integer)
- `videoId`: Unique video identifier (text)
- Vector embeddings for each video title

## Configuration

### Environment Variables
Create a `.env` file with:
```
WEAVIATE_HOST=localhost
WEAVIATE_PORT=8080
```

### Model Configuration
The default sentence transformer model is `all-mpnet-base-v2`. You can modify this in `WeaviateEmbedder.py`:
```python
MODEL_NAME = "all-mpnet-base-v2"  # Change to other models as needed
```

## Output Files

- `data/video_distance_analysis.png`: Main visualization graph
- Console output: Detailed statistics and channel rankings

## Dependencies

- **Core**: `numpy`, `matplotlib`, `scikit-learn`
- **Vector Database**: `weaviate-client>=4.0.0`
- **Embeddings**: `sentence-transformers>=2.2.0`, `torch>=1.9.0`
- **Utilities**: `requests`, `python-dotenv`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues and questions:
1. Check the existing issues
2. Create a new issue with detailed description
3. Include error messages and system information

---

**Note**: This tool requires a Weaviate instance running with YouTube video data. Make sure your database is properly set up before running the analysis. 