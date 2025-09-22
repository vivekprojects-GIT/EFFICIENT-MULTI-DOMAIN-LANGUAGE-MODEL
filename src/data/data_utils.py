import json
import os
from datasets import Dataset
from typing import List, Dict, Any


def load_qna_json(file_path: str) -> List[Dict[str, Any]]:
    """
    Load Q&A data from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing Q&A data
        
    Returns:
        List of dictionaries containing question and answer pairs
    """
    # Check if file exists, if not try relative to project root
    if not os.path.exists(file_path):
        # Try to find the file relative to project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        alternative_path = os.path.join(project_root, file_path)
        
        if os.path.exists(alternative_path):
            file_path = alternative_path
        else:
            raise FileNotFoundError(f"Data file not found at {file_path} or {alternative_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract the Q&A pairs from the "qna" array
    return data.get("qna", [])


def build_hf_dataset(qna_data: List[Dict[str, Any]]) -> Dataset:
    """
    Build a Hugging Face dataset from Q&A data with train/validation splits.
    
    Args:
        qna_data: List of dictionaries containing question and answer pairs
        
    Returns:
        Dataset: Hugging Face dataset with train/test splits
    """
    # Convert to Hugging Face dataset format
    dataset = Dataset.from_list(qna_data)
    
    # Split into train and test sets (80/20 split)
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    
    return train_test_split