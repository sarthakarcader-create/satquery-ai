"""
Annotation Loader for SatQuery AI
===================================

Loads and processes the BigEarthNet.txt question-answer annotations.
These are the text queries that users ask about satellite images.

Dataset structure:
  - 9.5M total annotations (we use ~6,281 from our 300-pair prototype)
  - 4 task types: binary, mcq, bounding_box, captioning
  - 11 question categories: presence, area, count, adjacency, etc.
  - Each annotation links to a specific S1/S2 image pair

Example queries:
  - Binary: "Is there a water body in the image?" → "yes"
  - MCQ: "Which land cover is present?" → "A) Forest B) Water C) Urban"
  - Bounding Box: "Locate the water body" → "[0.2, 0.4, 0.8, 0.6]"
  - Captioning: "Describe the scene" → "A forested area with a river..."
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ============================================================
# Data Classes
# ============================================================

@dataclass
class Annotation:
    """A single question-answer annotation for a satellite image pair."""
    id: int
    patch_id: str          # S2 patch identifier
    s1_name: str           # S1 patch identifier
    pair_id: int           # Which of our 300 pairs this belongs to
    dataset_split: str     # train / validation / test
    
    # The question and answer
    question: str          # Natural language question
    answer: str            # The correct answer
    
    # Task metadata
    task_type: str         # binary, mcq, bounding_box, captioning
    category: str          # presence, area, count, adjacency, point, reference, etc.
    
    # Geographic metadata
    latitude: float = 0.0
    longitude: float = 0.0
    country: str = ""
    season: str = ""
    climate_zone: str = ""


@dataclass
class ImagePair:
    """A pair of S1/S2 images with their annotations."""
    pair_id: int
    s1_name: str
    s2_name: str           # patch_id
    split: str             # train / validation / test
    
    annotations: List[Annotation] = None
    
    def __post_init__(self):
        if self.annotations is None:
            self.annotations = []


# ============================================================
# Loading Functions
# ============================================================

def load_annotations(parquet_path: Path) -> pd.DataFrame:
    """
    Load the full BigEarthNet.txt annotations from parquet.
    
    This gives us the complete dataset with ~9.5M rows.
    For the prototype, we'll filter to our 300 pairs.
    """
    print(f"Loading annotations from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df):,} annotations")
    print(f"  Columns: {df.columns.tolist()}")
    return df


def load_prototype_annotations(
    annotations_path: Path = Path("/Users/adityaupadhyaya/Desktop/satquery-ai/data/processed/prototype_annotations.parquet"),
    pairs_path: Path = Path("/Users/adityaupadhyaya/Desktop/satquery-ai/data/processed/prototype_pairs.csv")
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load our prototype subset (300 pairs, ~6,281 annotations).
    
    Returns:
        (annotations_df, pairs_df)
    """
    annotations = pd.read_parquet(annotations_path)
    pairs = pd.read_csv(pairs_path)
    
    print(f"Prototype annotations: {len(annotations):,}")
    print(f"Prototype pairs: {len(pairs):,}")
    print(f"\nTask type distribution:")
    print(annotations["type"].value_counts().to_string())
    print(f"\nSplit distribution:")
    print(annotations["dataset_split"].value_counts().to_string())
    
    return annotations, pairs


def get_annotations_for_pair(
    annotations: pd.DataFrame,
    pair_id: int
) -> List[Annotation]:
    """Get all annotations for a specific image pair."""
    pair_annotations = annotations[annotations["pair_id"] == pair_id]
    
    result = []
    for _, row in pair_annotations.iterrows():
        ann = Annotation(
            id=row["ID"],
            patch_id=row["patch_id"],
            s1_name=row["s1_name"],
            pair_id=row["pair_id"],
            dataset_split=row["dataset_split"],
            question=row["input"],
            answer=row["output"],
            task_type=row["type"],
            category=row["category"],
            latitude=row.get("latitude", 0.0),
            longitude=row.get("longitude", 0.0),
            country=row.get("country", ""),
            season=row.get("season", ""),
            climate_zone=row.get("climate_zone", ""),
        )
        result.append(ann)
    
    return result


def get_annotations_by_type(
    annotations: pd.DataFrame,
    task_type: str
) -> pd.DataFrame:
    """Filter annotations by task type."""
    return annotations[annotations["type"] == task_type].copy()


def get_annotations_by_split(
    annotations: pd.DataFrame,
    split: str
) -> pd.DataFrame:
    """Filter annotations by dataset split."""
    return annotations[annotations["dataset_split"] == split].copy()


# ============================================================
# Task-Specific Processors
# ============================================================

def process_binary_answer(answer: str) -> int:
    """Convert yes/no answer to binary label (1=yes, 0=no)."""
    return 1 if answer.strip().lower() in ["yes", "true", "1"] else 0


def process_mcq_answer(answer: str, choices: List[str] = None) -> int:
    """
    Convert MCQ answer to index.
    If choices provided, find matching index. Otherwise, return 0/1.
    """
    if choices:
        for i, choice in enumerate(choices):
            if answer.strip().lower() == choice.strip().lower():
                return i
    return 0 if answer.strip().lower() in ["a", "first", "1"] else 1


def process_bbox_answer(answer: str) -> Optional[List[float]]:
    """
    Parse bounding box answer.
    
    Format: "[x1 y1, x2 y2]" or "[x1, y1, x2, y2]"
    Returns: [x1, y1, x2, y2] as floats in [0, 1]
    """
    try:
        # Remove brackets and parse
        clean = answer.strip().replace("[", "").replace("]", "")
        # Handle various separators
        clean = clean.replace(",", " ").replace("  ", " ")
        values = [float(v) for v in clean.split() if v]
        
        if len(values) == 4:
            return values
    except (ValueError, IndexError):
        pass
    return None


def process_caption_answer(answer: str) -> str:
    """Clean up caption text."""
    return answer.strip()


# ============================================================
# Summary Statistics
# ============================================================

def print_dataset_summary(annotations: pd.DataFrame):
    """Print a comprehensive summary of the dataset."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    
    print(f"\nTotal annotations: {len(annotations):,}")
    print(f"Unique pairs: {annotations['pair_id'].nunique()}")
    
    print("\n--- Task Types ---")
    for task_type, count in annotations["type"].value_counts().items():
        pct = count / len(annotations) * 100
        print(f"  {task_type:15s}: {count:>8,} ({pct:.1f}%)")
    
    print("\n--- Question Categories ---")
    for cat, count in annotations["category"].value_counts().items():
        pct = count / len(annotations) * 100
        print(f"  {cat:15s}: {count:>8,} ({pct:.1f}%)")
    
    print("\n--- Dataset Splits ---")
    for split, count in annotations["dataset_split"].value_counts().items():
        pct = count / len(annotations) * 100
        print(f"  {split:15s}: {count:>8,} ({pct:.1f}%)")
    
    print("\n--- Sample Questions ---")
    for task_type in ["binary", "mcq", "bounding box", "captioning"]:
        subset = annotations[annotations["type"] == task_type]
        if len(subset) > 0:
            sample = subset.iloc[0]
            print(f"\n  [{task_type.upper()}]")
            print(f"    Q: {sample['input'][:100]}...")
            print(f"    A: {sample['output'][:100]}...")


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    annotations, pairs = load_prototype_annotations()
    print_dataset_summary(annotations)
    
    # Show a specific pair
    print("\n" + "=" * 60)
    print("SAMPLE PAIR: ID 194")
    print("=" * 60)
    pair_anns = get_annotations_for_pair(annotations, 194)
    for ann in pair_anns[:5]:
        print(f"\n  [{ann.task_type}] {ann.category}")
        print(f"  Q: {ann.question}")
        print(f"  A: {ann.answer}")
