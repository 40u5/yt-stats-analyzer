"""
YouTube Video Distance Analysis

This script analyzes the relationship between subscriber count and 
average distance between videos for each YouTuber, creating a 
logarithmic scatter plot visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from pathlib import Path
from WeaviateEmbedder import WeaviateEmbedder
from sklearn.metrics.pairwise import cosine_distances


class VideoDistanceAnalyzer:
    """Analyzes video distances and creates visualizations."""
    
    def __init__(self):
        """Initialize the analyzer with Weaviate connection."""
        self.embedder = WeaviateEmbedder()
            
    def calculate_average_distance(self, embeddings: List[List[float]]) -> float:
        """Calculate the average cosine distance between all pairs of videos."""
        if len(embeddings) < 2:
            return 0.0
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings)
        
        # Calculate cosine distances between all pairs
        distances = cosine_distances(embeddings_array)
        
        # Get upper triangle (excluding diagonal) to avoid counting same pairs twice
        upper_triangle = np.triu(distances, k=1)
        
        # Calculate mean of non-zero values
        non_zero_distances = upper_triangle[upper_triangle > 0]
        
        if len(non_zero_distances) == 0:
            return 0.0
            
        return float(np.mean(non_zero_distances))


    def analyze_all_channels(self) -> List[Tuple[str, int, float]]:
        """Analyze all channels and return (channel_id, subscriber_count, avg_distance)."""
        try:
            collection = self.embedder.client.collections.get("YouTubeVideo")
            
            # Group videos by channel using regular dict
            channel_data = {}
            
            # Fetch all videos with their embeddings
            for obj in collection.iterator(
                include_vector=True,
                return_properties=["channelId", "subscriberCount", "title"],
                cache_size=1000
            ):
                channel_id = obj.properties.get("channelId")
                subscriber_count = obj.properties.get("subscriberCount", 0)
                
                # Extract vector properly - it might be in different formats
                vector = None
                if hasattr(obj, 'vector') and obj.vector is not None:
                    if isinstance(obj.vector, dict):
                        # If vector is a dict, try to get the 'default' key
                        vector = obj.vector.get('default', obj.vector)
                    elif isinstance(obj.vector, list):
                        vector = obj.vector
                    else:
                        print(f"Unexpected vector type: {type(obj.vector)} for channel {channel_id}")
                        continue
                
                if channel_id and vector and isinstance(vector, list):
                    # Initialize channel data if not exists
                    if channel_id not in channel_data:
                        channel_data[channel_id] = {
                            'videos': [],
                            'subscriber_count': subscriber_count
                        }
                    
                    # Add video embedding to channel
                    channel_data[channel_id]['videos'].append(vector)
            
            print(f"Found {len(channel_data)} channels to analyze")
            
            # Calculate average distances for each channel
            results = []
            for channel_id, data in channel_data.items():
                embeddings = data['videos']
                subscriber_count = data['subscriber_count']
                
                if len(embeddings) >= 2:
                    avg_distance = self.calculate_average_distance(embeddings)
                    results.append((channel_id, subscriber_count, avg_distance))
                    print(f"Channel {channel_id}: {len(embeddings)} videos, avg distance: {avg_distance:.4f}")
                elif subscriber_count == 0:
                    print(f"Channel {channel_id}: 0 subscribers")
                elif len(embeddings) < 2:
                    print(f"Channel {channel_id}: Only {len(embeddings)} video(s), skipping analysis")
            
            # Sort by average distance (ascending - lower distance means more similar content)
            results.sort(key=lambda x: x[2])
            
            return results
            
        except Exception as e:
            print(f"Error analyzing channels: {e}")
            return []


    def create_visualization(self, results: List[Tuple[str, int, float]]):
        """Create a scatter plot with logarithmic subscriber count."""
        if not results:
            print("No data to visualize")
            return
        
        # Extract data
        subscriber_counts = [r[1] for r in results]
        avg_distances = [r[2] for r in results]
        
        # Set up the plot
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot with logarithmic x-axis
        plt.scatter(subscriber_counts, avg_distances, alpha=0.6, s=50)
        
        # Set logarithmic scale for x-axis
        plt.xscale('log')
        
        # Add labels and title
        plt.xlabel('Subscriber Count (log scale)', fontsize=12)
        plt.ylabel('Average Distance Between Videos', fontsize=12)
        plt.title('YouTube Channel Analysis: Subscriber Count vs Video Similarity', fontsize=14, fontweight='bold')
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Add some statistics as text
        avg_distance = np.mean(avg_distances)
        std_distance = np.std(avg_distances)
        plt.text(0.02, 0.98, f'Mean Distance: {avg_distance:.4f}\nStd Distance: {std_distance:.4f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add trend line with data validation
        if len(results) > 1:
            # Filter out zero or negative subscriber counts for trend line
            valid_data = [(sub, dist) for sub, dist in zip(subscriber_counts, avg_distances) 
                        if sub > 0 and np.isfinite(dist)]
            
            if len(valid_data) > 1:
                valid_subscribers = [x[0] for x in valid_data]
                valid_distances = [x[1] for x in valid_data]
                
                # Log-transform subscriber counts for trend line
                log_subscribers = np.log10(valid_subscribers)
                
                # Additional check for finite values
                if np.all(np.isfinite(log_subscribers)) and np.all(np.isfinite(valid_distances)):
                    z = np.polyfit(log_subscribers, valid_distances, 1)
                    p = np.poly1d(z)
                    
                    # Plot trend line
                    x_trend = np.logspace(np.log10(min(valid_subscribers)), 
                                        np.log10(max(valid_subscribers)), 100)
                    y_trend = p(np.log10(x_trend))
                    plt.plot(x_trend, y_trend, "r--", alpha=0.8, 
                            label=f'Trend line (slope: {z[0]:.4f})')
                    plt.legend()
                else:
                    print("Warning: Unable to create trend line due to invalid data")
            else:
                print("Warning: Insufficient valid data for trend line")
        
        # Save the plot to a file
        output_path = Path("data/video_distance_analysis.png")
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Graph saved to: {output_path}")
        
        # Display the plot
        plt.show()
    
    
    def print_summary_statistics(self, results: List[Tuple[str, int, float]]):
        """Print summary statistics of the analysis."""
        if not results:
            return
            
        subscriber_counts = [r[1] for r in results]
        avg_distances = [r[2] for r in results]
        
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        print(f"Total channels analyzed: {len(results)}")
        print(f"Subscriber count range: {min(subscriber_counts):,} - {max(subscriber_counts):,}")
        print(f"Average distance range: {min(avg_distances):.4f} - {max(avg_distances):.4f}")
        print(f"Mean average distance: {np.mean(avg_distances):.4f}")
        print(f"Median average distance: {np.median(avg_distances):.4f}")
        print(f"Standard deviation: {np.std(avg_distances):.4f}")
        
        # Find channels with highest and lowest distances
        sorted_by_distance = sorted(results, key=lambda x: x[2], reverse=True)
        
        print(f"\nTop 5 channels with highest video diversity (distance):")
        for i, (channel_id, subs, distance) in enumerate(sorted_by_distance[:5], 1):
            print(f"  {i}. Channel {channel_id}: {distance:.4f} (distance), {subs:,} subscribers")
        
        print(f"\nTop 5 channels with lowest video diversity (distance):")
        for i, (channel_id, subs, distance) in enumerate(sorted_by_distance[-5:], 1):
            print(f"  {i}. Channel {channel_id}: {distance:.4f} (distance), {subs:,} subscribers")
    
    def close(self):
        """Close the embedder connection."""
        if hasattr(self, 'embedder'):
            self.embedder.close()

    def test_weaviate_connection(self):
        """Test if Weaviate is accessible and has data."""
        try:
            collection = self.embedder.client.collections.get("YouTubeVideo")
            total_count = collection.aggregate.over_all(total_count=True).total_count
            print(f"Weaviate connection successful. Total videos: {total_count}")
            
            # Get a sample of records to see the structure
            sample = list(collection.iterator(include_vector=True, return_properties=["channelId", "title"]))
            print(f"Sample record structure: {sample[0].properties if sample else 'No records'}")
            
            return True
        except Exception as e:
            print(f"Weaviate connection failed: {e}")
            return False


def main():
    """Main execution function."""
    print("Starting YouTube Video Distance Analysis...")
    
    analyzer = None
    try:
        analyzer = VideoDistanceAnalyzer()
        
        # Test connection first
        if not analyzer.test_weaviate_connection():
            print("Cannot connect to Weaviate. Please ensure it's running.")
            return
        
        # Analyze all channels
        results = analyzer.analyze_all_channels()
        
        if results:
            # Create visualization
            analyzer.create_visualization(results)
        else:
            print("No valid data found for analysis")
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise
    finally:
        if analyzer:
            analyzer.close()
            print("Analysis completed and resources cleaned up")


if __name__ == "__main__":
    main() 