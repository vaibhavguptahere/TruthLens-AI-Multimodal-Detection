"""
TruthLens AI - FastAPI REST Server
Exposes endpoints for multimodal content verification, explainability, and health check.
"""

from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.inference.pipeline import TruthLensPipeline

app = FastAPI(
    title="TruthLens AI API",
    description="Multimodal NLP & Deepfake Verification Engine",
    version="1.0.0"
)

# Enable CORS for local React interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (loaded once on startup)
pipeline: Optional[TruthLensPipeline] = None


@app.on_event("startup")
def startup_event():
    global pipeline
    print("Initializing TruthLens AI inference pipeline...")
    pipeline = TruthLensPipeline()
    print("TruthLens AI pipeline initialized and ready for requests.")


@app.get("/")
def root():
    return {
        "service": "TruthLens AI Multimodal Verification API",
        "status": "online",
        "docs_url": "/docs"
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None
    }


@app.post("/api/verify")
async def verify_endpoint(
    image: UploadFile = File(...),
    caption: str = Form(...)
):
    """
    Accepts an uploaded image and text caption.
    Returns 3-class prediction (REAL, FAKE, UNCERTAIN), confidence,
    cross-modal consistency analysis, Grad-CAM heatmap, and text saliency attributions.
    """
    global pipeline
    if pipeline is None:
        pipeline = TruthLensPipeline()

    if not caption or len(caption.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text caption must not be empty.")

    try:
        image_bytes = await image.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded image is empty.")

        # Run full multimodal verification
        result = pipeline.verify(image_input=image_bytes, caption_text=caption)
        return JSONResponse(status_code=200, content=result)

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=f"Image validation error: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline error: {str(e)}")
