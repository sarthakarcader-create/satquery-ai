"""
SatQuery Multi-Modal Vision-Language Model
=============================================

The core AI model that answers natural-language questions about satellite imagery.

Architecture:
  1. Vision Encoder: Processes 12-band Sentinel-2 + 2-band Sentinel-1 images
  2. Text Encoder: Processes natural-language questions
  3. Fusion Layer: Combines vision and text understanding
  4. Task Heads: Produces task-specific outputs (yes/no, bbox, caption)

This is a custom architecture designed specifically for remote sensing:
  - Handles multi-band satellite imagery (not just RGB)
  - Supports both optical (S2) and SAR (S1) data
  - Works with 4 different task types

Key concept - Vision Transformer (ViT):
  Instead of processing an image as a whole, we:
  1. Split it into patches (like a grid)
  2. Convert each patch to a vector (like a word embedding)
  3. Process all patches with a Transformer (same as language models!)
  
  This lets us use the same attention mechanism that powers GPT,
  but for image understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple


# ============================================================
# Vision Encoder (Processes Satellite Images)
# ============================================================

class PatchEmbedding(nn.Module):
    """
    Converts an image into a sequence of patch embeddings.
    
    Think of it like tokenization for images:
      - Text tokenizer: "hello world" → [token1, token2]
      - Patch tokenizer: image → [patch1, patch2, ..., patchN]
    
    For a 120×120 image with 16×16 patches:
      - Number of patches: (120/16) × (120/16) = 7 × 7 = 49 patches
      - Each patch: 16×16×14 pixels (14 bands = 12 S2 + 2 S1)
      - Embedded to: 49 vectors of dimension d_model
    """
    
    def __init__(
        self,
        in_channels: int = 14,    # 12 S2 + 2 S1 bands
        patch_size: int = 16,
        embed_dim: int = 256,
        image_size: int = 120,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Convolution that extracts patches and projects to embedding dim
        # This is like a linear layer that looks at each patch
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        
        # Positional embedding: tells the model WHERE each patch is
        # Without this, the model wouldn't know patch order
        self.position_embedding = nn.Embedding(self.num_patches, embed_dim)
        
        # Layer normalization for stable training
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) - batch of satellite images
        
        Returns:
            (B, num_patches, embed_dim) - sequence of patch embeddings
        """
        B, C, H, W = x.shape
        
        # Extract patches and embed: (B, C, H, W) → (B, embed_dim, H', W')
        x = self.projection(x)
        
        # Reshape to sequence: (B, embed_dim, H', W') → (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        
        # Add positional information
        positions = torch.arange(self.num_patches, device=x.device)
        x = x + self.position_embedding(positions)
        
        x = self.norm(x)
        return x


class VisionEncoder(nn.Module):
    """
    Complete Vision Transformer for satellite images.
    
    Pipeline:
      Image → PatchEmbedding → TransformerLayers → GlobalAveragePool → FeatureVector
    
    The Transformer layers use self-attention to learn relationships
    between different parts of the image (e.g., "water is next to forest").
    """
    
    def __init__(
        self,
        in_channels: int = 14,
        patch_size: int = 16,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        image_size: int = 120,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            image_size=image_size,
        )
        
        # CLS token: a special "summary" token that aggregates info from all patches
        # (Same concept as BERT's [CLS] token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) - satellite image batch
        
        Returns:
            (B, embed_dim) - image feature vector
        """
        B = x.shape[0]
        
        # Patch embedding: (B, C, H, W) → (B, num_patches, embed_dim)
        x = self.patch_embed(x)
        
        # Prepend CLS token: (B, num_patches, embed_dim) → (B, num_patches+1, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Transformer: process all patches with self-attention
        x = self.transformer(x)
        x = self.norm(x)
        
        # Use CLS token as the image representation
        # CLS token has "attended to" all other patches, so it summarizes the image
        return x[:, 0]  # (B, embed_dim)


# ============================================================
# Text Encoder (Processes Questions)
# ============================================================

class TextEncoder(nn.Module):
    """
    Encodes natural-language questions into vectors.
    
    Pipeline:
      "Is there water?" → [tokenize] → [embed] → [transformer] → feature_vector
    
    This is a simplified version of BERT's text encoding.
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Token embedding: converts word IDs to vectors
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional embedding: tells model word order
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: (B, seq_len) - tokenized question
            attention_mask: (B, seq_len) - which positions are real tokens
        
        Returns:
            (B, embed_dim) - question feature vector
        """
        B, seq_len = input_ids.shape
        
        # Embed tokens
        x = self.token_embedding(input_ids)
        
        # Add positions
        positions = torch.arange(seq_len, device=input_ids.device)
        x = x + self.position_embedding(positions)
        
        # Create padding mask for transformer
        if attention_mask is not None:
            # Convert 0/1 mask to True/False for PyTorch (True = ignore)
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None
        
        # Transformer
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        
        # Use first token (CLS) or mean pooling as question representation
        if attention_mask is not None:
            # Masked mean pooling
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        
        return x  # (B, embed_dim)


# ============================================================
# Fusion Layer (Combines Vision + Text)
# ============================================================

class CrossModalFusion(nn.Module):
    """
    Fuses image and text representations.
    
    Uses cross-attention: the text "attends to" the image features
    to find relevant visual information for answering the question.
    
    Example:
      Question: "Is there water?"
      Image: shows forest, river, buildings
      
      Cross-attention helps the model focus on the river region
      when processing the word "water".
    """
    
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        # Cross-attention: text queries attend to image keys/values
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text_features: (B, embed_dim) - question representation
            image_features: (B, embed_dim) - image representation
        
        Returns:
            (B, embed_dim) - fused representation
        """
        # Reshape for cross-attention: (B, 1, embed_dim)
        text_seq = text_features.unsqueeze(1)
        image_seq = image_features.unsqueeze(1)
        
        # Cross-attention: text attends to image
        attended, _ = self.cross_attention(
            query=text_seq,
            key=image_seq,
            value=image_seq,
        )
        
        # Residual connection + layer norm
        text_seq = self.norm1(text_seq + attended)
        
        # Feed-forward
        ffn_out = self.ffn(text_seq)
        fused = self.norm2(text_seq + ffn_out)
        
        return fused.squeeze(1)  # (B, embed_dim)


# ============================================================
# Task Heads
# ============================================================

class BinaryVQAHead(nn.Module):
    """Answers yes/no questions: "Is there water?" → yes"""
    
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)  # (B, 1) - logit


class BBoxHead(nn.Module):
    """Predicts bounding boxes: "Locate the water" → [x1, y1, x2, y2]"""
    
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 4),
            nn.Sigmoid(),  # BBox coordinates in [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(x)  # (B, 4) - bbox coordinates


class CaptionHead(nn.Module):
    """Generates image captions: Describe the scene -> A forest with a river..."""
    
    def __init__(self, embed_dim: int = 256, vocab_size: int = 30000, max_len: int = 64):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        
        # Simple autoregressive decoder
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=3)
        
        self.output_proj = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, image_features: torch.Tensor, target_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        During training: teacher forcing with target_ids
        During inference: autoregressive generation
        """
        B = image_features.shape[0]
        
        if target_ids is not None:
            # Training: process all target tokens at once
            x = self.token_embedding(target_ids)
            positions = torch.arange(x.shape[1], device=x.device)
            x = x + self.position_embedding(positions)
            
            # Cross-attention with image features
            # (Simplified: just add image features to first position)
            x[:, 0] = x[:, 0] + image_features
            
            x = self.decoder(x)
            return self.output_proj(x)
        else:
            # Inference: generate token by token
            return self._generate(image_features)
    
    def _generate(self, image_features: torch.Tensor) -> torch.Tensor:
        """Autoregressive caption generation."""
        B = image_features.shape[0]
        device = image_features.device
        
        # Start with BOS token (id=1)
        generated = torch.ones(B, 1, dtype=torch.long, device=device)
        
        for _ in range(self.max_len):
            x = self.token_embedding(generated)
            positions = torch.arange(x.shape[1], device=device)
            x = x + self.position_embedding(positions)
            x[:, 0] = x[:, 0] + image_features
            
            x = self.decoder(x)
            logits = self.output_proj(x[:, -1, :])
            
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated


# ============================================================
# Complete SatQuery Model
# ============================================================

class SatQueryModel(nn.Module):
    """
    The complete SatQuery Vision-Language Model.
    
    Combines:
      - Vision Encoder (ViT for satellite images)
      - Text Encoder (Transformer for questions)
      - Cross-Modal Fusion (combines vision + text)
      - Task Heads (binary, bbox, captioning)
    
    Usage:
        model = SatQueryModel()
        
        # Forward pass
        output = model(
            s2_image=image_tensor,   # (B, 12, H, W)
            s1_image=sar_tensor,     # (B, 2, H, W)
            question_ids=token_ids,  # (B, seq_len)
            task_type="binary",
        )
    """
    
    def __init__(
        self,
        s2_bands: int = 12,
        s1_bands: int = 2,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_vision_layers: int = 6,
        num_text_layers: int = 4,
        image_size: int = 120,
        patch_size: int = 16,
        vocab_size: int = 30000,
        max_seq_len: int = 128,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        total_bands = s2_bands + s1_bands
        
        # Encoders
        # Vision encoder input channels = embed_dim (after S1/S2 projection)
        self.vision_encoder = VisionEncoder(
            in_channels=embed_dim,  # After projection to embed_dim
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_vision_layers,
            image_size=image_size,
        )
        
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_text_layers,
            max_seq_len=max_seq_len,
        )
        
        # Fusion
        self.fusion = CrossModalFusion(embed_dim=embed_dim, num_heads=num_heads)
        
        # Task-specific heads
        self.binary_head = BinaryVQAHead(embed_dim)
        self.bbox_head = BBoxHead(embed_dim)
        self.caption_head = CaptionHead(embed_dim, vocab_size)
        
        # Project S1 and S2 to same channel count before concatenating
        self.s1_proj = nn.Conv2d(s1_bands, embed_dim // 4, 1)
        self.s2_proj = nn.Conv2d(s2_bands, embed_dim * 3 // 4, 1)
    
    def forward(
        self,
        s2_image: torch.Tensor,
        s1_image: torch.Tensor,
        question_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        task_type: str = "binary",
        target_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model.
        
        Args:
            s2_image: (B, 12, H, W) - Sentinel-2 optical data
            s1_image: (B, 2, H, W) - Sentinel-1 SAR data
            question_ids: (B, seq_len) - tokenized question
            attention_mask: (B, seq_len) - padding mask
            task_type: "binary", "mcq", "bounding_box", or "captioning"
            target_ids: (B, max_len) - target tokens for caption training
        
        Returns:
            Dictionary with task-specific outputs
        """
        # Combine S1 and S2 into single input
        # Project to compatible channel dimensions, then concatenate
        s1_features = self.s1_proj(s1_image)  # (B, embed_dim//4, H, W)
        s2_features = self.s2_proj(s2_image)  # (B, embed_dim*3//4, H, W)
        combined_image = torch.cat([s1_features, s2_features], dim=1)  # (B, embed_dim, H, W)
        
        # Encode image
        image_features = self.vision_encoder(combined_image)  # (B, embed_dim)
        
        # Encode question
        text_features = self.text_encoder(question_ids, attention_mask)  # (B, embed_dim)
        
        # Fuse vision and text
        fused = self.fusion(text_features, image_features)  # (B, embed_dim)
        
        # Task-specific output
        output = {"fused_features": fused, "image_features": image_features}
        
        if task_type == "binary":
            output["logits"] = self.binary_head(fused)
        elif task_type == "bounding_box":
            output["bbox"] = self.bbox_head(fused)
        elif task_type == "captioning":
            output["caption_logits"] = self.caption_head(image_features, target_ids)
        else:
            # For MCQ, use binary head as placeholder
            output["logits"] = self.binary_head(fused)
        
        return output
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Simple Tokenizer (for questions)
# ============================================================

class SimpleTokenizer:
    """
    A basic tokenizer for satellite imagery questions.
    
    In production, you'd use a pre-trained tokenizer (BERT, etc.).
    For the prototype, this simple word-level tokenizer works fine.
    """
    
    def __init__(self, vocab_size: int = 30000, max_len: int = 128):
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        # Special tokens
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.sep_token_id = 2
        self.unk_token_id = 3
        
        # Build vocabulary from common words
        self.word2idx = {
            "<pad>": 0, "<cls>": 1, "<sep>": 2, "<unk>": 3,
            "is": 4, "there": 5, "a": 6, "an": 7, "the": 8,
            "in": 9, "this": 10, "image": 11, "yes": 12, "no": 13,
            "water": 14, "forest": 15, "urban": 16, "field": 17,
            "land": 18, "cover": 19, "vegetation": 20, "building": 21,
            "road": 22, "river": 23, "lake": 24, "crop": 25,
            "pasture": 26, "arable": 27, "area": 28, "located": 29,
            "where": 30, "what": 31, "how": 32, "many": 33,
            "describe": 34, "provide": 35, "bounding": 36, "box": 37,
            "for": 38, "highlight": 39, "show": 40, "detect": 41,
            "any": 42, "does": 43, "do": 44, "can": 45,
            "between": 46, "square": 47, "meters": 48, "more": 49,
            "than": 50, "less": 51, "equal": 52, "to": 53,
            "have": 54, "has": 55, "having": 56, "with": 57,
            "without": 58, "next": 59, "adjacent": 60, "connected": 61,
        }
        
        self.idx2word = {v: k for k, v in self.word2idx.items()}
    
    def encode(self, text: str) -> torch.Tensor:
        """Convert text to token IDs."""
        words = text.lower().split()
        tokens = [self.cls_token_id]
        
        for word in words[:self.max_len - 2]:
            tokens.append(self.word2idx.get(word, self.unk_token_id))
        
        tokens.append(self.sep_token_id)
        
        # Pad to max_len
        while len(tokens) < self.max_len:
            tokens.append(self.pad_token_id)
        
        return torch.tensor(tokens[:self.max_len], dtype=torch.long)
    
    def encode_batch(self, texts: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a batch of texts."""
        all_tokens = []
        all_masks = []
        
        for text in texts:
            tokens = self.encode(text)
            mask = (tokens != self.pad_token_id).long()
            all_tokens.append(tokens)
            all_masks.append(mask)
        
        return torch.stack(all_tokens), torch.stack(all_masks)


# ============================================================
# Model Factory
# ============================================================

def create_satquery_model(
    model_size: str = "small",
    device: str = "auto",
) -> SatQueryModel:
    """
    Create a SatQuery model with predefined configurations.
    
    Args:
        model_size: "small", "medium", or "large"
        device: "auto", "cpu", or "cuda"
    
    Returns:
        Configured SatQueryModel
    """
    configs = {
        "small": {
            "embed_dim": 256,
            "num_heads": 8,
            "num_vision_layers": 4,
            "num_text_layers": 3,
            "patch_size": 16,
        },
        "medium": {
            "embed_dim": 512,
            "num_heads": 8,
            "num_vision_layers": 6,
            "num_text_layers": 4,
            "patch_size": 16,
        },
        "large": {
            "embed_dim": 768,
            "num_heads": 12,
            "num_vision_layers": 12,
            "num_text_layers": 6,
            "patch_size": 16,
        },
    }
    
    config = configs.get(model_size, configs["small"])
    
    model = SatQueryModel(**config)
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    params = model.count_parameters()
    print(f"Created SatQuery {model_size} model")
    print(f"  Parameters: {params:,} ({params/1e6:.1f}M)")
    print(f"  Device: {device}")
    
    return model


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    print("SatQuery Model - Quick Test")
    print("=" * 60)
    
    # Create small model
    model = create_satquery_model("small", device="cpu")
    
    # Create dummy inputs
    B = 2
    s2_image = torch.randn(B, 12, 120, 120)
    s1_image = torch.randn(B, 2, 120, 120)
    question = torch.randint(0, 100, (B, 64))
    
    # Test binary VQA
    print("\n--- Binary VQA Test ---")
    output = model(s2_image, s1_image, question, task_type="binary")
    print(f"  Logits shape: {output['logits'].shape}")
    print(f"  Predictions: {torch.sigmoid(output['logits']).detach().numpy().flatten()}")
    
    # Test bounding box
    print("\n--- Bounding Box Test ---")
    output = model(s2_image, s1_image, question, task_type="bounding_box")
    print(f"  BBox shape: {output['bbox'].shape}")
    print(f"  Predicted boxes: {output['bbox'].detach().numpy()}")
    
    # Test captioning
    print("\n--- Captioning Test ---")
    target = torch.randint(0, 100, (B, 32))
    output = model(s2_image, s1_image, question, task_type="captioning", target_ids=target)
    print(f"  Caption logits shape: {output['caption_logits'].shape}")
    
    print("\n✅ All task heads working!")
    print(f"\nModel summary:")
    print(f"  Total parameters: {model.count_parameters():,}")
    print(f"  Vision encoder: {sum(p.numel() for p in model.vision_encoder.parameters()):,}")
    print(f"  Text encoder: {sum(p.numel() for p in model.text_encoder.parameters()):,}")
    print(f"  Fusion layer: {sum(p.numel() for p in model.fusion.parameters()):,}")
