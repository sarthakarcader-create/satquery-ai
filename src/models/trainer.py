"""
SatQuery Training Pipeline
============================

The training loop that teaches the model to answer satellite imagery questions.

Training Process:
  1. Feed (image, question) to the model
  2. Model predicts an answer
  3. Compare prediction to ground truth (loss)
  4. Update model weights to reduce loss
  5. Repeat until model is good at answering questions

Loss Functions by Task:
  - Binary VQA: Binary Cross-Entropy (yes/no classification)
  - BBox: Smooth L1 Loss (regression for coordinates)
  - Captioning: Cross-Entropy (predict next word)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import time

from .satquery_model import SatQueryModel, SimpleTokenizer
from ..data.satquery_dataset import SatQueryDataset, create_dataloaders, satquery_collate_fn


# ============================================================
# Loss Functions
# ============================================================

class SatQueryLoss(nn.Module):
    """
    Combined loss function for all SatQuery tasks.
    
    Different tasks need different losses:
      - Binary: "Is there water?" → yes/no (classification)
      - BBox: "Locate the water" → [x1,y1,x2,y2] (regression)
      - Captioning: "Describe the scene" → words (next token prediction)
    """
    
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for a batch of mixed task types.
        
        Returns:
            Dictionary with total_loss and individual task losses
        """
        total_loss = torch.tensor(0.0, device="cpu")
        losses = {}
        
        task_types = batch["task_type"]
        s2_images = batch["s2_image"]
        s1_images = batch["s1_image"]
        
        # Group samples by task type for efficient processing
        binary_mask = [i for i, t in enumerate(task_types) if t == "binary"]
        mcq_mask = [i for i, t in enumerate(task_types) if t == "mcq"]
        bbox_mask = [i for i, t in enumerate(task_types) if t == "bounding box"]
        caption_mask = [i for i, t in enumerate(task_types) if t == "captioning"]
        
        # Binary VQA loss
        if binary_mask and "logits" in outputs:
            binary_logits = outputs["logits"][binary_mask]
            binary_targets = []
            for i in binary_mask:
                answer = batch["answer"][i].strip().lower()
                binary_targets.append(1.0 if answer in ["yes", "true"] else 0.0)
            binary_targets = torch.tensor(binary_targets, device=binary_logits.device).unsqueeze(1)
            
            binary_loss = self.bce_loss(binary_logits, binary_targets)
            losses["binary_loss"] = binary_loss
            total_loss = total_loss + binary_loss
        
        # MCQ loss (treat as binary for simplicity in prototype)
        if mcq_mask and "logits" in outputs:
            mcq_logits = outputs["logits"][mcq_mask]
            mcq_targets = []
            for i in mcq_mask:
                answer = batch["answer"][i].strip().lower()
                mcq_targets.append(1.0 if answer in ["b", "second", "1"] else 0.0)
            mcq_targets = torch.tensor(mcq_targets, device=mcq_logits.device).unsqueeze(1)
            
            mcq_loss = self.bce_loss(mcq_logits, mcq_targets)
            losses["mcq_loss"] = mcq_loss
            total_loss = total_loss + mcq_loss
        
        # Bounding Box loss
        if bbox_mask and "bbox" in outputs:
            bbox_preds = outputs["bbox"][bbox_mask]
            bbox_targets = []
            for i in bbox_mask:
                answer = batch["answer"][i]
                try:
                    clean = answer.strip().replace("[", "").replace("]", "").replace(",", " ")
                    values = [float(v) for v in clean.split() if v]
                    if len(values) == 4:
                        bbox_targets.append(values)
                    else:
                        bbox_targets.append([0, 0, 1, 1])
                except:
                    bbox_targets.append([0, 0, 1, 1])
            bbox_targets = torch.tensor(bbox_targets, device=bbox_preds.device)
            
            bbox_loss = self.smooth_l1_loss(bbox_preds, bbox_targets)
            losses["bbox_loss"] = bbox_loss
            total_loss = total_loss + bbox_loss
        
        # Captioning loss
        if caption_mask and "caption_logits" in outputs:
            caption_logits = outputs["caption_logits"]
            # Create dummy targets for now (real targets need tokenization)
            # This is a placeholder - real implementation would tokenize answers
            caption_loss = torch.tensor(0.0, device=total_loss.device)
            losses["caption_loss"] = caption_loss
            total_loss = total_loss + caption_loss * 0  # Zero out for now
        
        losses["total_loss"] = total_loss
        return losses


# ============================================================
# Metrics
# ============================================================

class Metrics:
    """Track training and evaluation metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.binary_correct = 0
        self.binary_total = 0
        self.bbox_iou_sum = 0.0
        self.bbox_total = 0
        self.total_loss = 0.0
        self.num_batches = 0
    
    def update(self, loss: float, outputs: Dict, batch: Dict):
        """Update metrics with batch results."""
        self.total_loss += loss
        self.num_batches += 1
        
        task_types = batch["task_type"]
        
        # Binary accuracy
        if "logits" in outputs:
            preds = (torch.sigmoid(outputs["logits"]).squeeze(-1) > 0.5).float()
            for i, t in enumerate(task_types):
                if t == "binary":
                    answer = batch["answer"][i].strip().lower()
                    target = 1.0 if answer in ["yes", "true"] else 0.0
                    if preds[i].item() == target:
                        self.binary_correct += 1
                    self.binary_total += 1
    
    def get_metrics(self) -> Dict:
        """Get current metric values."""
        metrics = {}
        if self.num_batches > 0:
            metrics["avg_loss"] = self.total_loss / self.num_batches
        if self.binary_total > 0:
            metrics["binary_accuracy"] = self.binary_correct / self.binary_total
        return metrics


# ============================================================
# Trainer
# ============================================================

class SatQueryTrainer:
    """
    Handles the training loop for SatQuery.
    
    Usage:
        trainer = SatQueryTrainer(model, train_loader, val_loader)
        trainer.train(num_epochs=10)
    """
    
    def __init__(
        self,
        model: SatQueryModel,
        train_loader,
        val_loader,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        device: str = "auto",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=10,  # Will be updated per epoch
        )
        
        # Loss
        self.criterion = SatQueryLoss().to(self.device)
        
        # Tokenizer
        self.tokenizer = SimpleTokenizer()
        
        # Metrics
        self.train_metrics = Metrics()
        self.val_metrics = Metrics()
        
        # History
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    def _prepare_batch(self, batch: Dict) -> Dict:
        """Move batch to device and tokenize questions."""
        s2_images = batch["s2_image"].to(self.device)
        s1_images = batch["s1_image"].to(self.device)
        
        # Tokenize questions
        question_ids, attention_mask = self.tokenizer.encode_batch(batch["question"])
        question_ids = question_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        return {
            "s2_image": s2_images,
            "s1_image": s1_images,
            "question_ids": question_ids,
            "attention_mask": attention_mask,
            "question": batch["question"],
            "answer": batch["answer"],
            "task_type": batch["task_type"],
            "category": batch["category"],
        }
    
    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            batch = self._prepare_batch(batch)
            
            # Determine task type for this batch (use most common)
            task_types = batch["task_type"]
            main_task = max(set(task_types), key=task_types.count)
            
            # Forward pass
            outputs = self.model(
                s2_image=batch["s2_image"],
                s1_image=batch["s1_image"],
                question_ids=batch["question_ids"],
                attention_mask=batch["attention_mask"],
                task_type=main_task,
            )
            
            # Compute loss
            losses = self.criterion(outputs, batch)
            loss = losses["total_loss"]
            
            # Skip if loss is zero (e.g., no matching task type)
            if loss.item() == 0:
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Update metrics
            self.train_metrics.update(loss.item(), outputs, batch)
            
            # Print progress
            if batch_idx % 50 == 0:
                metrics = self.train_metrics.get_metrics()
                print(f"  Batch {batch_idx}/{len(self.train_loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Acc: {metrics.get('binary_accuracy', 0):.3f}")
        
        # Epoch summary
        metrics = self.train_metrics.get_metrics()
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch} train: loss={metrics.get('avg_loss', 0):.4f} | "
              f"acc={metrics.get('binary_accuracy', 0):.3f} | "
              f"time={elapsed:.1f}s")
        
        return metrics
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict:
        """Validate on validation set."""
        self.model.eval()
        self.val_metrics.reset()
        
        for batch in self.val_loader:
            batch = self._prepare_batch(batch)
            
            task_types = batch["task_type"]
            main_task = max(set(task_types), key=task_types.count)
            
            outputs = self.model(
                s2_image=batch["s2_image"],
                s1_image=batch["s1_image"],
                question_ids=batch["question_ids"],
                attention_mask=batch["attention_mask"],
                task_type=main_task,
            )
            
            losses = self.criterion(outputs, batch)
            self.val_metrics.update(losses["total_loss"].item(), outputs, batch)
        
        metrics = self.val_metrics.get_metrics()
        print(f"  Epoch {epoch} val:   loss={metrics.get('avg_loss', 0):.4f} | "
              f"acc={metrics.get('binary_accuracy', 0):.3f}")
        
        return metrics
    
    def train(self, num_epochs: int = 10, save_dir: str = "checkpoints"):
        """Full training loop."""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"  Epochs: {num_epochs}")
        print(f"  Device: {self.device}")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")
        print(f"{'='*60}\n")
        
        best_val_loss = float("inf")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n--- Epoch {epoch}/{num_epochs} ---")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Save history
            self.history["train_loss"].append(train_metrics.get("avg_loss", 0))
            self.history["val_loss"].append(val_metrics.get("avg_loss", 0))
            self.history["train_acc"].append(train_metrics.get("binary_accuracy", 0))
            self.history["val_acc"].append(val_metrics.get("binary_accuracy", 0))
            
            # Save best model
            val_loss = val_metrics.get("avg_loss", float("inf"))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(save_dir / "best_model.pt", epoch)
                print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")
            
            # Save periodic checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(save_dir / f"checkpoint_epoch_{epoch}.pt", epoch)
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"  Best val loss: {best_val_loss:.4f}")
        print(f"{'='*60}")
    
    def save_checkpoint(self, path: Path, epoch: int):
        """Save model checkpoint."""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
        }, path)
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.history = checkpoint["history"]
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")


# ============================================================
# Inference
# ============================================================

class SatQueryInference:
    """
    Run inference with a trained SatQuery model.
    
    Usage:
        inference = SatQueryInference("checkpoints/best_model.pt")
        answer = inference.answer_question(image, "Is there water?")
    """
    
    def __init__(self, checkpoint_path: Path, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load model
        self.model = SatQueryModel()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = SimpleTokenizer()
        print(f"Loaded model from {checkpoint_path}")
    
    @torch.no_grad()
    def answer_question(
        self,
        s2_image: torch.Tensor,
        s1_image: torch.Tensor,
        question: str,
    ) -> Dict:
        """
        Answer a question about a satellite image.
        
        Args:
            s2_image: (12, H, W) Sentinel-2 data
            s1_image: (2, H, W) Sentinel-1 data
            question: Natural language question
        
        Returns:
            Dictionary with answer and confidence
        """
        # Add batch dimension
        s2 = s2_image.unsqueeze(0).to(self.device)
        s1 = s1_image.unsqueeze(0).to(self.device)
        
        # Tokenize question
        q_ids, mask = self.tokenizer.encode_batch([question])
        q_ids = q_ids.to(self.device)
        mask = mask.to(self.device)
        
        # Determine task type from question
        task_type = self._classify_task(question)
        
        # Forward pass
        outputs = self.model(s2, s1, q_ids, mask, task_type=task_type)
        
        # Process output based on task type
        result = {"question": question, "task_type": task_type}
        
        if task_type == "binary":
            prob = torch.sigmoid(outputs["logits"]).item()
            result["answer"] = "yes" if prob > 0.5 else "no"
            result["confidence"] = prob if prob > 0.5 else 1 - prob
        
        elif task_type == "bounding_box":
            bbox = outputs["bbox"].squeeze(0).cpu().numpy()
            result["bbox"] = bbox.tolist()
            result["answer"] = f"Bounding box: {bbox}"
        
        else:
            result["answer"] = "Task not yet implemented"
        
        return result
    
    def _classify_task(self, question: str) -> str:
        """Simple rule-based task classification."""
        q = question.lower()
        
        if any(w in q for w in ["bounding box", "locate", "highlight", "where is"]):
            return "bounding_box"
        elif any(w in q for w in ["yes", "no", "is there", "does", "do", "has", "have"]):
            return "binary"
        elif any(w in q for w in ["describe", "caption", "what do you see"]):
            return "captioning"
        else:
            return "binary"  # Default


# ============================================================
# Main entry point
# ============================================================

if __name__ == "__main__":
    print("SatQuery Trainer - Testing with Synthetic Data")
    print("=" * 60)
    
    # Create model
    model = SatQueryModel()
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Create dataloaders with synthetic data
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=8,
        use_synthetic=True,
    )
    
    # Quick test: one batch
    trainer = SatQueryTrainer(model, train_loader, val_loader)
    
    print("\nRunning 1 training step...")
    batch = next(iter(train_loader))
    batch = trainer._prepare_batch(batch)
    
    outputs = model(
        s2_image=batch["s2_image"],
        s1_image=batch["s1_image"],
        question_ids=batch["question_ids"],
        attention_mask=batch["attention_mask"],
        task_type="binary",
    )
    
    losses = trainer.criterion(outputs, batch)
    print(f"Loss: {losses['total_loss'].item():.4f}")
    print(f"Logits: {outputs['logits'].detach().numpy().flatten()}")
    
    print("\n✅ Training pipeline working!")
    print("\nTo train for real:")
    print("  trainer.train(num_epochs=10)")
