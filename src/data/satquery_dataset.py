"""
SatQuery PyTorch Dataset
=========================

This is the heart of the data pipeline — it combines:
  1. Satellite images (S1 + S2 bands)
  2. Natural language questions
  3. Answers (yes/no, choices, bounding boxes, captions)

into training samples that the model can learn from.

Key concept: How the Dataset class works
  ┌─────────────────────────────────────────────────────┐
  │  __init__: Load annotation file, build index         │
  │     ↓                                                │
  │  __len__: Return total number of samples             │
  │     ↓                                                │
  │  __getitem__(idx):                                    │
  │     1. Look up annotation by index                   │
  │     2. Load the corresponding S1 + S2 images         │
  │     3. Process the question text                     │
  │     4. Process the answer (depends on task type)     │
  │     5. Return (images, question, answer, metadata)   │
  └─────────────────────────────────────────────────────┘

This follows PyTorch's Dataset protocol:
  - __init__: setup
  - __len__: how many samples
  - __getitem__: get one sample
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Import our loaders
from .image_loader import load_s2_bands, load_s1_bands, S2_BAND_NAMES, S1_BAND_NAMES
from .annotation_loader import (
    load_prototype_annotations,
    get_annotations_for_pair,
    process_binary_answer,
    process_bbox_answer,
    Annotation
)


# ============================================================
# Sample Data Structure
# ============================================================

@dataclass
class SatQuerySample:
    """
    A single training/inference sample for SatQuery.
    
    This is what the model receives:
      - s2_image: Sentinel-2 optical data (12 bands)
      - s1_image: Sentinel-1 SAR data (2 bands)
      - question: The natural language query
      - answer: The ground truth answer
      - task_type: What kind of task (binary, mcq, bbox, captioning)
      - metadata: Additional info (pair_id, split, etc.)
    """
    s2_image: torch.Tensor    # (12, H, W)
    s1_image: torch.Tensor    # (2, H, W)
    question: str             # Natural language question
    answer: str               # Ground truth answer
    answer_tensor: torch.Tensor  # Processed answer (depends on task type)
    task_type: str            # binary, mcq, bounding_box, captioning
    category: str             # presence, area, count, etc.
    metadata: Dict[str, Any]  # Extra info


# ============================================================
# The Dataset Class
# ============================================================

class SatQueryDataset(Dataset):
    """
    PyTorch Dataset for SatQuery AI.
    
    Each sample is a (image, question, answer) triple from the
    BigEarthNet.txt dataset.
    
    Usage:
        dataset = SatQueryDataset(split="train")
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        for batch in loader:
            s2_images = batch["s2_image"]
            s1_images = batch["s1_image"]
            questions = batch["question"]
            answers = batch["answer_tensor"]
            # Feed to model...
    """
    
    def __init__(
        self,
        split: str = "train",
        s2_data_dir: Path = Path("/Users/adityaupadhyaya/Desktop/satquery-ai/data/raw/sentinel2"),
        s1_data_dir: Path = Path("/Users/adityaupadhyaya/Desktop/satquery-ai/data/raw/sentinel1"),
        annotations_path: Path = Path("/Users/adityaupadhyaya/Desktop/satquery-ai/data/processed/prototype_annotations.parquet"),
        pairs_path: Path = Path("/Users/adityaupadhyaya/Desktop/satquery-ai/data/processed/prototype_pairs.csv"),
        image_size: Tuple[int, int] = (120, 120),
        use_synthetic: bool = False,
    ):
        """
        Args:
            split: "train", "validation", or "test"
            s2_data_dir: Where Sentinel-2 patches are stored
            s1_data_dir: Where Sentinel-1 patches are stored
            annotations_path: Path to prototype_annotations.parquet
            pairs_path: Path to prototype_pairs.csv
            image_size: (H, W) to resize images to
            use_synthetic: If True, generate fake images (for testing)
        """
        self.split = split
        self.s2_data_dir = Path(s2_data_dir)
        self.s1_data_dir = Path(s1_data_dir)
        self.image_size = image_size
        self.use_synthetic = use_synthetic
        
        # Load annotations
        self.annotations, self.pairs = load_prototype_annotations(
            annotations_path, pairs_path
        )
        
        # Filter to requested split
        self.annotations = self.annotations[
            self.annotations["dataset_split"] == split
        ].reset_index(drop=True)
        
        print(f"\nSatQueryDataset initialized:")
        print(f"  Split: {split}")
        print(f"  Samples: {len(self.annotations):,}")
        print(f"  Image size: {image_size}")
        print(f"  Use synthetic: {use_synthetic}")
        
        # Build pair_id → image directory mapping
        self.pair_dirs = self._build_pair_dirs()
        
        # Check data availability
        if not use_synthetic:
            available = sum(1 for pid in self.annotations["pair_id"].unique() 
                          if pid in self.pair_dirs)
            print(f"  Pairs with images: {available}/{len(self.pair_dirs)}")
    
    def _build_pair_dirs(self) -> Dict[int, Dict[str, Path]]:
        """Build mapping from pair_id to S1/S2 directory paths."""
        pair_dirs = {}
        
        for _, row in self.pairs.iterrows():
            pair_id = row["pair_id"]
            s1_name = row["s1_name"]
            s2_name = row["patch_id"]
            
            # Find the actual directories (may be nested)
            s2_dir = self._find_patch_dir(self.s2_data_dir, s2_name)
            s1_dir = self._find_patch_dir(self.s1_data_dir, s1_name)
            
            pair_dirs[pair_id] = {
                "s2": s2_dir,
                "s1": s1_dir,
                "s1_name": s1_name,
                "s2_name": s2_name,
            }
        
        return pair_dirs
    
    def _find_patch_dir(self, base_dir: Path, patch_name: str) -> Optional[Path]:
        """Find the directory containing band files for a patch."""
        if self.use_synthetic:
            return None  # Will use synthetic data
        
        # Search recursively for the patch directory
        matches = list(base_dir.rglob(patch_name))
        for match in matches:
            if match.is_dir():
                return match
        
        return None
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> SatQuerySample:
        """
        Get a single training sample.
        
        Returns a SatQuerySample containing:
          - S2 image tensor (12, H, W)
          - S1 image tensor (2, H, W)
          - Question string
          - Answer tensor (processed for the task type)
          - Task type and metadata
        """
        row = self.annotations.iloc[idx]
        
        # Get pair info
        pair_id = row["pair_id"]
        pair_info = self.pair_dirs.get(pair_id, {})
        
        # Load images (or generate synthetic ones)
        if self.use_synthetic or pair_info.get("s2") is None:
            s2_image = self._generate_synthetic_s2()
            s1_image = self._generate_synthetic_s1()
        else:
            try:
                s2_image = torch.from_numpy(
                    load_s2_bands(pair_info["s2"], target_size=self.image_size)
                )
                s1_image = torch.from_numpy(
                    load_s1_bands(pair_info["s1"], target_size=self.image_size)
                )
            except Exception as e:
                # Fallback to synthetic if loading fails
                print(f"  Warning: Failed to load images for pair {pair_id}: {e}")
                s2_image = self._generate_synthetic_s2()
                s1_image = self._generate_synthetic_s1()
        
        # Get question and answer
        question = str(row["input"])
        answer = str(row["output"])
        task_type = str(row["type"])
        category = str(row["category"])
        
        # Process answer based on task type
        answer_tensor = self._process_answer(answer, task_type)
        
        # Build metadata
        metadata = {
            "pair_id": pair_id,
            "s1_name": row["s1_name"],
            "s2_name": row["patch_id"],
            "latitude": row.get("latitude", 0.0),
            "longitude": row.get("longitude", 0.0),
            "country": row.get("country", ""),
            "season": row.get("season", ""),
            "climate_zone": row.get("climate_zone", ""),
            "original_answer": answer,
        }
        
        return SatQuerySample(
            s2_image=s2_image,
            s1_image=s1_image,
            question=question,
            answer=answer,
            answer_tensor=answer_tensor,
            task_type=task_type,
            category=category,
            metadata=metadata,
        )
    
    def _process_answer(self, answer: str, task_type: str) -> torch.Tensor:
        """
        Convert the answer string into a tensor based on task type.
        
        - Binary: tensor([1.0]) for yes, tensor([0.0]) for no
        - MCQ: tensor([0]) or tensor([1])
        - Bounding box: tensor([x1, y1, x2, y2])
        - Captioning: the raw string (tokenized later by the model)
        """
        if task_type == "binary":
            label = process_binary_answer(answer)
            return torch.tensor([label], dtype=torch.float32)
        
        elif task_type == "mcq":
            # Simple: 0 for first option, 1 for second
            label = 0 if answer.strip().lower() in ["a", "first", "1", "0"] else 1
            return torch.tensor([label], dtype=torch.float32)
        
        elif task_type == "bounding box":
            bbox = process_bbox_answer(answer)
            if bbox:
                return torch.tensor(bbox, dtype=torch.float32)
            else:
                return torch.tensor([0, 0, 1, 1], dtype=torch.float32)
        
        elif task_type == "captioning":
            # Return as string — model will tokenize
            return answer
        
        else:
            return torch.tensor([0], dtype=torch.float32)
    
    def _generate_synthetic_s2(self) -> torch.Tensor:
        """Generate synthetic Sentinel-2 data for testing."""
        h, w = self.image_size
        # Simulate spectral bands with realistic patterns
        synthetic = torch.randn(12, h, w) * 0.3 + 0.5
        synthetic = torch.clamp(synthetic, 0, 1)
        return synthetic.float()
    
    def _generate_synthetic_s1(self) -> torch.Tensor:
        """Generate synthetic Sentinel-1 data for testing."""
        h, w = self.image_size
        synthetic = torch.randn(2, h, w) * 0.2 + 0.3
        synthetic = torch.clamp(synthetic, 0, 1)
        return synthetic.float()


# ============================================================
# Data Collation
# ============================================================

def satquery_collate_fn(batch: List[SatQuerySample]) -> Dict[str, Any]:
    """
    Custom collate function for DataLoader.
    
    This handles the fact that different samples may have different
    task types (binary vs captioning) and thus different answer formats.
    
    Returns a dictionary with batched tensors and lists of strings.
    """
    return {
        "s2_image": torch.stack([s.s2_image for s in batch]),
        "s1_image": torch.stack([s.s1_image for s in batch]),
        "question": [s.question for s in batch],
        "answer": [s.answer for s in batch],
        "answer_tensor": batch,  # Keep individual tensors (different shapes)
        "task_type": [s.task_type for s in batch],
        "category": [s.category for s in batch],
        "metadata": [s.metadata for s in batch],
    }


# ============================================================
# DataLoader Factory
# ============================================================

def create_dataloaders(
    batch_size: int = 8,
    num_workers: int = 0,
    image_size: Tuple[int, int] = (120, 120),
    use_synthetic: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = SatQueryDataset(
        split="train",
        image_size=image_size,
        use_synthetic=use_synthetic,
    )
    val_ds = SatQueryDataset(
        split="validation",
        image_size=image_size,
        use_synthetic=use_synthetic,
    )
    test_ds = SatQueryDataset(
        split="test",
        image_size=image_size,
        use_synthetic=use_synthetic,
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=satquery_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=satquery_collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=satquery_collate_fn,
    )
    
    return train_loader, val_loader, test_loader


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    print("SatQuery Dataset - Testing with Synthetic Data")
    print("=" * 60)
    
    # Create dataset with synthetic images (no download needed)
    dataset = SatQueryDataset(split="train", use_synthetic=True)
    
    print(f"\nDataset length: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  S2 image shape: {sample.s2_image.shape}")
    print(f"  S1 image shape: {sample.s1_image.shape}")
    print(f"  Question: {sample.question[:80]}...")
    print(f"  Answer: {sample.answer}")
    print(f"  Task type: {sample.task_type}")
    print(f"  Category: {sample.category}")
    print(f"  Answer tensor: {sample.answer_tensor}")
    
    # Test DataLoader
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=satquery_collate_fn
    )
    
    batch = next(iter(loader))
    print(f"\nBatch shapes:")
    print(f"  S2 images: {batch['s2_image'].shape}")
    print(f"  S1 images: {batch['s1_image'].shape}")
    print(f"  Questions: {len(batch['question'])} items")
    print(f"  Task types: {batch['task_type']}")
    
    print("\n✅ Dataset working! Ready to connect to real images when download completes.")
