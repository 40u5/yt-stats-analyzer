"""
YouTube Data Embedding Script

Creates embeddings for YouTube video titles using a lightweight sentence transformer model.
Designed to work with less than 8GB VRAM.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

from sentence_transformers import SentenceTransformer
import torch


class YouTubeDataEmbedder:
    """
    Embeds YouTube video data using sentence transformers.
    
    Uses lightweight models to stay within 8GB VRAM constraint.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder with a lightweight model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        print(f"Loading model: {model_name}")
        
        # Load the model
        self.model = SentenceTransformer(model_name)
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print(f"Model loaded on: {self.device}")
        print(f"Model dimension: {self.model.get_sentence_embedding_dimension()}")
        
        # Check VRAM usage
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1024**3
            print(f"VRAM used: {vram_used:.2f} GB")
    
    def load_data(self, file_path: str) -> Dict[str, List[List]]:
        """
        Load YouTube data from JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dictionary with channel IDs as keys and video data as values
        """
        print(f"Loading data from: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} channels")
        return data
    
    def prepare_texts(self, data: Dict[str, List[List]]) -> Tuple[List[str], List[str], List[int]]:
        """
        Prepare texts for embedding from the YouTube data.
        
        Args:
            data: YouTube data dictionary
            
        Returns:
            Tuple of (texts, channel_ids, view_counts)
        """
        texts = []
        channel_ids = []
        view_counts = []
        
        for channel_id, videos in data.items():
            for title, views in videos:
                texts.append(title)
                channel_ids.append(channel_id)
                view_counts.append(views)
        
        print(f"Prepared {len(texts)} video titles for embedding")
        return texts, channel_ids, view_counts
    
    def create_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Create embeddings for the given texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings
        """
        print(f"Creating embeddings for {len(texts)} texts...")
        
        # Create embeddings in batches to manage memory
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch_texts, 
                convert_to_numpy=True,
                show_progress_bar=True
            )
            embeddings.append(batch_embeddings)
            
            # Print progress
            progress = min(i + batch_size, len(texts))
            print(f"Processed {progress}/{len(texts)} texts")
        
        # Concatenate all batches
        all_embeddings = np.vstack(embeddings)
        print(f"Created embeddings with shape: {all_embeddings.shape}")
        
        return all_embeddings
    
    def save_embeddings(
        self, 
        embeddings: np.ndarray, 
        channel_ids: List[str], 
        view_counts: List[int],
        output_dir: str = "data"
    ) -> str:
        """
        Save embeddings and metadata to files.
        
        Args:
            embeddings: Numpy array of embeddings
            channel_ids: List of channel IDs
            view_counts: List of view counts
            output_dir: Output directory
            
        Returns:
            Path to the saved embeddings file
        """
        # Ensure output directory exists
        Path(output_dir).mkdir(exist_ok=True)
        
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save embeddings
        embeddings_file = Path(output_dir) / f"embeddings_{timestamp}.npy"
        np.save(embeddings_file, embeddings)
        
        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "embedding_dimension": embeddings.shape[1],
            "num_videos": len(embeddings),
            "channel_ids": channel_ids,
            "view_counts": view_counts,
            "created_at": timestamp,
            "device": str(self.device)
        }
        
        metadata_file = Path(output_dir) / f"embeddings_metadata_{timestamp}.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Saved embeddings to: {embeddings_file}")
        print(f"Saved metadata to: {metadata_file}")
        
        return str(embeddings_file)
    
    def process_file(self, input_file: str, output_dir: str = "data", batch_size: int = 32) -> str:
        """
        Complete pipeline: load data, create embeddings, and save results.
        
        Args:
            input_file: Path to input JSON file
            output_dir: Output directory for embeddings
            batch_size: Batch size for processing
            
        Returns:
            Path to the saved embeddings file
        """
        print("=" * 60)
        print("YouTube Data Embedding Pipeline")
        print("=" * 60)
        
        # Load data
        data = self.load_data(input_file)
        
        # Prepare texts
        texts, channel_ids, view_counts = self.prepare_texts(data)
        
        # Create embeddings
        embeddings = self.create_embeddings(texts, batch_size)
        
        # Save results
        output_file = self.save_embeddings(embeddings, channel_ids, view_counts, output_dir)
        
        print("=" * 60)
        print("Embedding pipeline completed successfully!")
        print(f"Total videos processed: {len(embeddings)}")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        print(f"Output saved to: {output_file}")
        print("=" * 60)
        
        return output_file


def main():
    """Main execution function."""
    # Configuration
    input_file = "data/results.json"
    output_dir = "data"
    model_name = "all-mpnet-base-v2"  # (~420MB, ~4-5GB VRAM)
    batch_size = 16  # Reduced batch size for model
    
    # Alternative models in 4-7GB VRAM range:
    # "all-mpnet-base-v2"      # ~420MB, ~4-5GB VRAM - Good balance of quality/speed
    # "all-MiniLM-L12-v2"      # ~120MB, ~3-4GB VRAM - Lightweight but good quality
    # "multi-qa-mpnet-base-dot-v1"  # ~420MB, ~4-5GB VRAM - Optimized for similarity
    # "paraphrase-multilingual-mpnet-base-v2"  # ~420MB, ~4-5GB VRAM - Multilingual
    
    print(f"Using model: {model_name}")
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Initialize embedder
        embedder = YouTubeDataEmbedder(model_name)
        
        # Process the data
        output_file = embedder.process_file(input_file, output_dir, batch_size)
        
        print(f"\n✅ Embedding completed successfully!")
        print(f"📁 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error during embedding: {e}")
        raise


if __name__ == "__main__":
    main()