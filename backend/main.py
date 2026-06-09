"""Factory Vision Anomaly Detection System — FastAPI application entry point."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from model.efficientnet import get_available_part_types
from routes.parts import router as parts_router
from routes.predict import router as predict_router
from services.predictor import get_loaded_parts

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Factory Vision Anomaly Detection API",
    description=(
        "Multi-part anomaly detection — EfficientNet-B0 feature extraction and "
        "nearest-neighbour memory bank scoring, one model per part type. "
        "Trained exclusively on OK images; no defect examples required."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)
app.include_router(parts_router)


@app.get("/health", summary="System health and model status")
async def health_check() -> dict:
    """
    Return system health and per-part model readiness.

    Response fields
    ---------------
    status          : "healthy" if the API process is running.
    available_parts : part types with a trained model on disk.
    loaded_parts    : part types currently in the in-process cache.
    """
    return {
        "status": "healthy",
        "available_parts": get_available_part_types(),
        "loaded_parts": get_loaded_parts(),
    }
