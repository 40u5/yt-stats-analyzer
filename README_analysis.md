# YouTube Video Distance Analysis

This script analyzes the relationship between YouTube channel subscriber counts and the average distance between videos within each channel.

## What it does

The script creates a scatter plot showing:
- **X-axis**: Subscriber count (logarithmic scale)
- **Y-axis**: Average distance between videos (using cosine distance on embeddings)

This helps visualize whether channels with more subscribers tend to have more diverse or more similar video content.

## Prerequisites

1. **Weaviate Database**: Make sure Weaviate is running locally on port 8080
2. **Data**: Ensure you have scraped data in `data/results.json`
3. **Dependencies**: Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the analysis script:

```bash
python analyze_video_distances.py
```

## Output

The script will:
1. Load your scraped YouTube data
2. Query Weaviate for video embeddings for each channel
3. Calculate average distances between videos within each channel
4. Generate a scatter plot saved as `data/subscriber_vs_distance_analysis.png`
5. Print summary statistics including:
   - Total channels analyzed
   - Distance ranges and statistics
   - Top channels with highest/lowest video diversity

## Interpretation

- **Higher distance values** = More diverse video content (videos are less similar to each other)
- **Lower distance values** = More similar video content (videos are more alike)
- **Trend line** = Shows the overall relationship between subscriber count and content diversity

## Example Insights

This analysis can reveal patterns like:
- Do larger channels tend to have more diverse content?
- Are niche channels more focused on similar content?
- What's the relationship between audience size and content variety?

## Troubleshooting

- **Weaviate connection error**: Make sure Weaviate is running on localhost:8080
- **No data found**: Ensure you've run the scraper first and have data in `data/results.json`
- **Missing embeddings**: Make sure videos were properly embedded into Weaviate during scraping 