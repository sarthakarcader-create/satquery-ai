"""
SatQuery AI - Web Application
===============================

Interactive web interface for SatQuery AI.
Upload satellite images, ask questions, and get intelligent responses.

Run with:
    python app.py

Or with a specific port:
    python app.py --port 7860
"""

import gradio as gr
import torch
import numpy as np
from pathlib import Path
import sys
import json
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


from src.models.satquery_model import SatQueryModel, SimpleTokenizer
from src.agents.controller import SatQueryController, classify_query


# ============================================================
# Initialize Components
# ============================================================

print("Initializing SatQuery AI...")

# Create model (will use random weights for demo)
model = SatQueryModel()
print(f"Model loaded: {model.count_parameters():,} parameters")

# Create tokenizer
tokenizer = SimpleTokenizer()

# Create controller
controller = SatQueryController(model=model, tokenizer=tokenizer)
print("Controller initialized")


# ============================================================
# Core Processing Function
# ============================================================

def process_satquery(
    s2_image,       # Sentinel-2 image (numpy array from upload)
    s1_image,       # Sentinel-1 image (numpy array from upload)
    query,          # User's question
    use_synthetic,  # Use synthetic data for demo
):
    """
    Main processing function for the Gradio interface.
    
    Takes user inputs, processes through the controller, returns response.
    """
    if not query or not query.strip():
        return {
            "answer": "Please enter a question.",
            "confidence": 0.0,
            "task_type": "none",
            "execution_trace": "No query provided.",
        }
    
    # Convert images to tensors
    s2_tensor = None
    s1_tensor = None
    
    if s2_image is not None:
        # Gradio gives us a numpy array (H, W, C) or (H, W)
        img = np.array(s2_image)
        if img.ndim == 2:
            # Grayscale → fake 12-band
            img = np.stack([img] * 12, axis=-1)
        elif img.shape[-1] == 3:
            # RGB → repeat to fill 12 bands
            img = np.concatenate([img, img, img, img], axis=-1)[:, :, :12]
        
        # (H, W, C) → (C, H, W)
        s2_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    
    if s1_image is not None:
        img = np.array(s1_image)
        if img.ndim == 2:
            img = np.stack([img, img], axis=-1)
        elif img.shape[-1] == 3:
            img = img[:, :, :2]
        
        s1_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    
    # If no real images and synthetic mode, create fake ones
    if use_synthetic or (s2_tensor is None and s1_tensor is None):
        s2_tensor = torch.randn(12, 120, 120) * 0.3 + 0.5
        s1_tensor = torch.randn(2, 120, 120) * 0.2 + 0.3
    
    # Process through controller
    result = controller.process_query(
        query=query,
        s2_image=s2_tensor,
        s1_image=s1_tensor,
    )
    
    # Format output as 4 separate values for Gradio textboxes
    if result.get("success"):
        answer = result["answer"]
        confidence = f"{result['confidence']:.1%}"
        task_type = result["task_type"]
        trace = json.dumps(result["trace"], indent=2)
    else:
        answer = f"Error: {result.get('errors', ['Unknown error'])}"
        confidence = "0%"
        task_type = "error"
        trace = "Processing failed."
    
    return answer, confidence, task_type, trace


# ============================================================
# Visualization
# ============================================================

def create_visual_overlay(s2_image, result):
    """Create a visualization with bounding box overlay if applicable."""
    if s2_image is None:
        return None
    
    img = np.array(s2_image).copy()
    
    # If we have a bounding box, draw it
    bbox = result.get("bbox")
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        h, w = img.shape[:2]
        
        # Convert normalized coords to pixel coords
        x1_px, y1_px = int(x1 * w), int(y1 * h)
        x2_px, y2_px = int(x2 * w), int(y2 * h)
        
        # Draw rectangle (simple approach - just set pixels)
        if img.ndim == 3:
            img[y1_px:y1_px+2, x1_px:x2_px] = [255, 0, 0]  # Top edge
            img[y2_px-2:y2_px, x1_px:x2_px] = [255, 0, 0]  # Bottom edge
            img[y1_px:y2_px, x1_px:x1_px+2] = [255, 0, 0]  # Left edge
            img[y1_px:y2_px, x2_px-2:x2_px] = [255, 0, 0]  # Right edge
    
    return img


# ============================================================
# Gradio Interface
# ============================================================

def build_ui():
    """Build the Gradio web interface."""
    
    with gr.Blocks(
        title="SatQuery AI - Satellite Image Question Answering",
        theme=gr.themes.Soft(),
    ) as demo:
        
        gr.Markdown("""
        # 🛰️ SatQuery AI
        ### Intelligent Satellite Image Analysis
        
        Upload satellite images (Sentinel-2 optical or Sentinel-1 SAR) and ask natural language questions.
        The AI will analyze the imagery and provide evidence-grounded responses.
        
        **Supported Tasks:**
        - 🟢 **Binary VQA**: "Is there water in this image?" → Yes/No
        - 📦 **Bounding Box**: "Highlight the forested area" → Detection
        - 📝 **Captioning**: "Describe this scene" → Description
        """)
        
        with gr.Row():
            # Left column: Inputs
            with gr.Column(scale=1):
                gr.Markdown("### 📸 Input Images")
                
                s2_input = gr.Image(
                    label="Sentinel-2 (Optical)",
                    type="numpy",
                    height=200,
                )
                
                s1_input = gr.Image(
                    label="Sentinel-1 (SAR)",
                    type="numpy",
                    height=200,
                )
                
                query_input = gr.Textbox(
                    label="❓ Your Question",
                    placeholder="e.g., Is there water in this image?",
                    lines=2,
                )
                
                synthetic_toggle = gr.Checkbox(
                    label="Use synthetic data (demo mode)",
                    value=True,
                    info="Use random data if no real images uploaded",
                )
                
                submit_btn = gr.Button(
                    "🚀 Analyze",
                    variant="primary",
                    size="lg",
                )
            
            # Right column: Outputs
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Results")
                
                answer_output = gr.Textbox(
                    label="Answer",
                    lines=3,
                    interactive=False,
                )
                
                with gr.Row():
                    confidence_output = gr.Textbox(
                        label="Confidence",
                        interactive=False,
                    )
                    task_output = gr.Textbox(
                        label="Task Type",
                        interactive=False,
                    )
                
                trace_output = gr.Textbox(
                    label="Execution Trace",
                    lines=8,
                    interactive=False,
                )
        
        # Example queries
        gr.Markdown("### 💡 Example Queries")
        gr.Examples(
            examples=[
                ["Is there water in this image?"],
                ["Highlight the forested area"],
                ["Describe the land cover in this scene"],
                ["Are there any buildings visible?"],
                ["Locate the agricultural fields"],
                ["What type of terrain is shown?"],
            ],
            inputs=query_input,
        )
        
        # Connect submit button
        submit_btn.click(
            fn=process_satquery,
            inputs=[s2_input, s1_input, query_input, synthetic_toggle],
            outputs=[answer_output, confidence_output, task_output, trace_output],
        )
        
        # Also submit on Enter
        query_input.submit(
            fn=process_satquery,
            inputs=[s2_input, s1_input, query_input, synthetic_toggle],
            outputs=[answer_output, confidence_output, task_output, trace_output],
        )
    
    return demo


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SatQuery AI Web Interface")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"SatQuery AI - Starting Web Interface")
    print(f"  URL: http://{args.host}:{args.port}")
    print(f"  Share: {args.share}")
    print(f"{'='*60}\n")
    
    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
