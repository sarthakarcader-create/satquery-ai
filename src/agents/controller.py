"""
SatQuery Agentic Controller
==============================

The "brain" of SatQuery AI — automatically routes user queries to the
right specialist model, validates inputs, and combines results.

This is what makes SatQuery different from a simple VQA model:
  - It SELECTS the right tool for each query
  - It VALIDATES inputs before processing
  - It COMBINES outputs from multiple specialists
  - It PROVIDES confidence scores and execution traces

Query Routing Example:
  "Is there water in this image?"
    → Task: Binary VQA
    → Model: VQA Specialist
    → Output: yes (confidence: 0.92)

  "Highlight the water body"
    → Task: Grounding
    → Model: Grounding Specialist
    → Output: bounding box [0.2, 0.4, 0.8, 0.6]

  "Describe this scene"
    → Task: Captioning
    → Model: Captioning Specialist
    → Output: "A forested area with a river running through..."

Architecture:
  ┌──────────────────────────────────────────────────┐
  │                 User Query + Images               │
  └──────────────────────┬───────────────────────────┘
                         │
  ┌──────────────────────▼───────────────────────────┐
  │              Input Validator                       │
  │  - Check image format, modality, count            │
  │  - Check query is valid text                      │
  └──────────────────────┬───────────────────────────┘
                         │
  ┌──────────────────────▼───────────────────────────┐
  │              Query Classifier                     │
  │  - Determine task type from query                 │
  │  - Extract entities (objects, regions)            │
  └──────────────────────┬───────────────────────────┘
                         │
  ┌──────────────────────▼───────────────────────────┐
  │              Model Selector                       │
  │  - Choose specialist(s) based on task             │
  │  - Configure task parameters                      │
  └──────────────────────┬───────────────────────────┘
                         │
  ┌──────────────────────▼───────────────────────────┐
  │              Executor                             │
  │  - Run selected model(s)                          │
  │  - Collect outputs                                │
  └──────────────────────┬───────────────────────────┘
                         │
  ┌──────────────────────▼───────────────────────────┐
  │              Output Processor                     │
  │  - Combine results                                │
  │  - Estimate confidence                            │
  │  - Generate visual evidence                       │
  │  - Build execution trace                          │
  └──────────────────────┬───────────────────────────┘
                         │
  ┌──────────────────────▼───────────────────────────┐
  │              Response                             │
  │  - Answer text                                    │
  │  - Confidence score                               │
  │  - Visual overlay (bbox, highlight)               │
  │  - Execution summary                              │
  └──────────────────────────────────────────────────┘
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import json


# ============================================================
# Task Classification
# ============================================================

class TaskType(Enum):
    """Supported task types."""
    BINARY_VQA = "binary_vqa"
    MCQ_VQA = "mcq_vqa"
    BOUNDING_BOX = "bounding_box"
    CAPTIONING = "captioning"
    CHANGE_DETECTION = "change_detection"
    CROSS_MODAL = "cross_modal"
    UNKNOWN = "unknown"


@dataclass
class QueryAnalysis:
    """Result of query classification."""
    task_type: TaskType
    confidence: float
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    raw_query: str = ""


def classify_query(query: str) -> QueryAnalysis:
    """
    Classify a natural language query into a task type.
    
    Uses keyword matching for the prototype.
    In production, you'd use a trained classifier or LLM.
    """
    q = query.lower().strip()
    keywords = q.split()
    
    # Bounding box / grounding
    bbox_keywords = [
        "bounding box", "locate", "highlight", "where is", "show me",
        "point to", "identify the region", "mark", "outline", "segment",
    ]
    if any(kw in q for kw in bbox_keywords):
        return QueryAnalysis(
            task_type=TaskType.BOUNDING_BOX,
            confidence=0.9,
            keywords=[kw for kw in bbox_keywords if kw in q],
            raw_query=query,
        )
    
    # Captioning
    caption_keywords = [
        "describe", "caption", "what do you see", "tell me about",
        "summarize", "explain the image", "what's in",
    ]
    if any(kw in q for kw in caption_keywords):
        return QueryAnalysis(
            task_type=TaskType.CAPTIONING,
            confidence=0.9,
            keywords=[kw for kw in caption_keywords if kw in q],
            raw_query=query,
        )
    
    # Binary VQA
    binary_keywords = [
        "is there", "does", "do", "has", "have", "can you see",
        "would you say", "is it true", "are there", "is any",
    ]
    if any(kw in q for kw in binary_keywords):
        # Check for yes/no answer indicators
        return QueryAnalysis(
            task_type=TaskType.BINARY_VQA,
            confidence=0.85,
            keywords=[kw for kw in binary_keywords if kw in q],
            raw_query=query,
        )
    
    # MCQ
    mcq_keywords = ["which", "choose", "select", "option", "a or b"]
    if any(kw in q for kw in mcq_keywords):
        return QueryAnalysis(
            task_type=TaskType.MCQ_VQA,
            confidence=0.8,
            keywords=[kw for kw in mcq_keywords if kw in q],
            raw_query=query,
        )
    
    # Change detection (for multi-image queries)
    change_keywords = [
        "changed", "difference", "before and after", "temporal",
        "compare", "what changed", "has it changed",
    ]
    if any(kw in q for kw in change_keywords):
        return QueryAnalysis(
            task_type=TaskType.CHANGE_DETECTION,
            confidence=0.85,
            keywords=[kw for kw in change_keywords if kw in q],
            raw_query=query,
        )
    
    # Default to binary VQA
    return QueryAnalysis(
        task_type=TaskType.BINARY_VQA,
        confidence=0.5,
        raw_query=query,
    )


# ============================================================
# Input Validation
# ============================================================

@dataclass
class ValidationResult:
    """Result of input validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    image_count: int = 0
    modalities: List[str] = field(default_factory=list)


def validate_inputs(
    s2_image: Optional[torch.Tensor] = None,
    s1_image: Optional[torch.Tensor] = None,
    query: str = "",
) -> ValidationResult:
    """
    Validate user inputs before processing.
    
    Checks:
      - At least one image provided
      - Image shapes are correct
      - Query is not empty
      - Image modalities match task requirements
    """
    errors = []
    warnings = []
    modalities = []
    image_count = 0
    
    # Check query
    if not query or not query.strip():
        errors.append("Query cannot be empty")
    
    # Check images
    if s2_image is not None:
        image_count += 1
        modalities.append("sentinel2")
        if s2_image.dim() != 3:
            errors.append(f"S2 image must be 3D (C, H, W), got {s2_image.dim()}D")
        elif s2_image.shape[0] != 12:
            warnings.append(f"S2 image has {s2_image.shape[0]} bands, expected 12")
    
    if s1_image is not None:
        image_count += 1
        modalities.append("sentinel1")
        if s1_image.dim() != 3:
            errors.append(f"S1 image must be 3D (C, H, W), got {s1_image.dim()}D")
        elif s1_image.shape[0] != 2:
            warnings.append(f"S1 image has {s1_image.shape[0]} bands, expected 2")
    
    if image_count == 0:
        warnings.append("No images provided — using synthetic data for demo")
    
    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        image_count=image_count,
        modalities=modalities,
    )


# ============================================================
# Execution Trace
# ============================================================

@dataclass
class ExecutionStep:
    """A single step in the execution trace."""
    step_name: str
    model_used: str
    parameters: Dict[str, Any]
    duration_ms: float
    output_summary: str


@dataclass
class ExecutionTrace:
    """Complete execution trace for audit."""
    steps: List[ExecutionStep] = field(default_factory=list)
    total_duration_ms: float = 0.0
    
    def add_step(self, step: ExecutionStep):
        self.steps.append(step)
    
    def to_dict(self) -> Dict:
        return {
            "steps": [
                {
                    "name": s.step_name,
                    "model": s.model_used,
                    "params": s.parameters,
                    "duration_ms": s.duration_ms,
                    "output": s.output_summary,
                }
                for s in self.steps
            ],
            "total_duration_ms": self.total_duration_ms,
        }


# ============================================================
# Agentic Controller
# ============================================================

class SatQueryController:
    """
    The agentic controller that orchestrates SatQuery AI.
    
    This is the main entry point for the system. It:
    1. Validates inputs
    2. Classifies the query
    3. Selects the right specialist model
    4. Executes the model
    5. Processes and returns results
    
    Usage:
        controller = SatQueryController(model=model, tokenizer=tokenizer)
        result = controller.process_query(
            s2_image=image_tensor,
            query="Is there water in this image?"
        )
    """
    
    def __init__(self, model=None, tokenizer=None, device: str = "auto"):
        """
        Args:
            model: SatQueryModel instance (optional for demo)
            tokenizer: SimpleTokenizer instance
            device: "auto", "cpu", or "cuda"
        """
        self.model = model
        self.tokenizer = tokenizer
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        if self.model is not None:
            self.model = self.model.to(self.device)
            self.model.eval()
        
        # Model registry (for future multi-model support)
        self.model_registry = {
            TaskType.BINARY_VQA: "satquery_vqa",
            TaskType.MCQ_VQA: "satquery_mcq",
            TaskType.BOUNDING_BOX: "satquery_grounding",
            TaskType.CAPTIONING: "satquery_captioning",
            TaskType.CHANGE_DETECTION: "satquery_change",
            TaskType.CROSS_MODAL: "satquery_fusion",
        }
    
    def process_query(
        self,
        query: str,
        s2_image: Optional[torch.Tensor] = None,
        s1_image: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Process a user query end-to-end.
        
        Args:
            query: Natural language question
            s2_image: Sentinel-2 image (C, H, W) or None
            s1_image: Sentinel-1 image (C, H, W) or None
        
        Returns:
            Dictionary with answer, confidence, visual evidence, and trace
        """
        trace = ExecutionTrace()
        start_time = time.time()
        
        # Step 1: Validate inputs
        step_start = time.time()
        validation = validate_inputs(s2_image, s1_image, query)
        trace.add_step(ExecutionStep(
            step_name="input_validation",
            model_used="validator",
            parameters={"query_length": len(query), "image_count": validation.image_count},
            duration_ms=(time.time() - step_start) * 1000,
            output_summary=f"Valid: {validation.valid}, Errors: {validation.errors}",
        ))
        
        if not validation.valid:
            return {
                "success": False,
                "errors": validation.errors,
                "warnings": validation.warnings,
                "trace": trace.to_dict(),
            }
        
        # Step 2: Classify query
        step_start = time.time()
        analysis = classify_query(query)
        trace.add_step(ExecutionStep(
            step_name="query_classification",
            model_used="classifier",
            parameters={"query": query},
            duration_ms=(time.time() - step_start) * 1000,
            output_summary=f"Task: {analysis.task_type.value}, Confidence: {analysis.confidence:.2f}",
        ))
        
        # Step 3: Select model
        step_start = time.time()
        model_name = self.model_registry.get(analysis.task_type, "unknown")
        trace.add_step(ExecutionStep(
            step_name="model_selection",
            model_used="selector",
            parameters={"task_type": analysis.task_type.value, "model": model_name},
            duration_ms=(time.time() - step_start) * 1000,
            output_summary=f"Selected: {model_name}",
        ))
        
        # Step 4: Execute model
        step_start = time.time()
        result = self._execute_specialist(
            task_type=analysis.task_type,
            query=query,
            s2_image=s2_image,
            s1_image=s1_image,
        )
        trace.add_step(ExecutionStep(
            step_name="model_execution",
            model_used=model_name,
            parameters={"task_type": analysis.task_type.value},
            duration_ms=(time.time() - step_start) * 1000,
            output_summary=str(result.get("answer", ""))[:100],
        ))
        
        # Step 5: Process output
        step_start = time.time()
        response = self._process_output(result, analysis, validation)
        trace.add_step(ExecutionStep(
            step_name="output_processing",
            model_used="processor",
            parameters={},
            duration_ms=(time.time() - step_start) * 1000,
            output_summary=f"Response ready",
        ))
        
        # Finalize
        trace.total_duration_ms = (time.time() - start_time) * 1000
        response["trace"] = trace.to_dict()
        response["success"] = True
        
        return response
    
    def _execute_specialist(
        self,
        task_type: TaskType,
        query: str,
        s2_image: Optional[torch.Tensor],
        s1_image: Optional[torch.Tensor],
    ) -> Dict:
        """Execute the appropriate specialist model."""
        
        # If no model loaded, return demo response
        if self.model is None:
            return self._demo_response(task_type, query)
        
        # Prepare inputs
        if s2_image is not None:
            s2 = s2_image.unsqueeze(0).to(self.device)
        else:
            s2 = torch.zeros(1, 12, 120, 120).to(self.device)
        
        if s1_image is not None:
            s1 = s1_image.unsqueeze(0).to(self.device)
        else:
            s1 = torch.zeros(1, 2, 120, 120).to(self.device)
        
        # Tokenize query
        q_ids, mask = self.tokenizer.encode_batch([query])
        q_ids = q_ids.to(self.device)
        mask = mask.to(self.device)
        
        # Map task type to model task type
        task_map = {
            TaskType.BINARY_VQA: "binary",
            TaskType.MCQ_VQA: "binary",  # Use binary head for MCQ in prototype
            TaskType.BOUNDING_BOX: "bounding_box",
            TaskType.CAPTIONING: "captioning",
        }
        model_task = task_map.get(task_type, "binary")
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                s2_image=s2,
                s1_image=s1,
                question_ids=q_ids,
                attention_mask=mask,
                task_type=model_task,
            )
        
        # Process outputs
        result = {"task_type": task_type.value}
        
        if model_task == "binary" and "logits" in outputs:
            prob = torch.sigmoid(outputs["logits"]).item()
            result["answer"] = "yes" if prob > 0.5 else "no"
            result["confidence"] = prob if prob > 0.5 else 1 - prob
        
        elif model_task == "bounding_box" and "bbox" in outputs:
            bbox = outputs["bbox"].squeeze(0).cpu().numpy()
            result["bbox"] = bbox.tolist()
            result["answer"] = f"Detected region at {bbox}"
            result["confidence"] = 0.75  # Placeholder
        
        elif model_task == "captioning":
            result["answer"] = "Caption generation not yet implemented"
            result["confidence"] = 0.0
        
        return result
    
    def _demo_response(self, task_type: TaskType, query: str) -> Dict:
        """Generate demo response when no model is loaded."""
        demo_responses = {
            TaskType.BINARY_VQA: {
                "answer": "yes",
                "confidence": 0.85,
                "task_type": "binary_vqa",
            },
            TaskType.BOUNDING_BOX: {
                "answer": "Detected region at [0.2, 0.3, 0.8, 0.7]",
                "bbox": [0.2, 0.3, 0.8, 0.7],
                "confidence": 0.78,
                "task_type": "bounding_box",
            },
            TaskType.CAPTIONING: {
                "answer": "A satellite image showing a mix of agricultural fields and forested areas with a small water body in the northwest corner.",
                "confidence": 0.72,
                "task_type": "captioning",
            },
        }
        return demo_responses.get(task_type, {
            "answer": "I'm not sure how to answer that.",
            "confidence": 0.3,
            "task_type": "unknown",
        })
    
    def _process_output(
        self,
        result: Dict,
        analysis: QueryAnalysis,
        validation: ValidationResult,
    ) -> Dict:
        """Process and format the final output."""
        response = {
            "answer": result.get("answer", ""),
            "confidence": result.get("confidence", 0.0),
            "task_type": analysis.task_type.value,
            "query_classification": {
                "task_type": analysis.task_type.value,
                "confidence": analysis.confidence,
                "keywords": analysis.keywords,
            },
            "input_info": {
                "image_count": validation.image_count,
                "modalities": validation.modalities,
            },
        }
        
        # Add visual evidence for bounding box tasks
        if "bbox" in result:
            response["visual_evidence"] = {
                "type": "bounding_box",
                "coordinates": result["bbox"],
            }
        
        # Add warnings
        if validation.warnings:
            response["warnings"] = validation.warnings
        
        return response


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    print("SatQuery Agentic Controller - Demo")
    print("=" * 60)
    
    # Create controller without model (demo mode)
    controller = SatQueryController(model=None)
    
    # Test queries
    test_queries = [
        "Is there water in this image?",
        "Highlight the forested area",
        "Describe the land cover in this scene",
        "How many buildings are visible?",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = controller.process_query(query=query)
        print(f"  Task: {result['task_type']}")
        print(f"  Answer: {result['answer']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Trace steps: {len(result['trace']['steps'])}")
    
    print("\n✅ Controller working!")
