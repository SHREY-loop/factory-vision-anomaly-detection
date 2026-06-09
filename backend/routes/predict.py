"""Prediction API routes — anomaly detection inspection endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from schemas.prediction import PredictionResponse
from services.predictor import predict as run_prediction
from training.config import MAX_UPLOAD_BYTES, validate_part_type

router = APIRouter(tags=["prediction"])

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


async def validate_image_file(file: UploadFile = File(...)) -> UploadFile:
    """Validate uploaded file type before processing."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    extension = (
        "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    )
    content_type = (file.content_type or "").lower()

    if extension not in ALLOWED_EXTENSIONS and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Accepted formats: jpg, jpeg, png.",
        )
    return file


@router.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(
    file: UploadFile = Depends(validate_image_file),
    part_type: str = Query(
        ...,
        min_length=1,
        description="Part type to run inference against (e.g. 'part_A', 'part_B', 'part_C').",
    ),
) -> PredictionResponse:
    """
    Accept a multipart image upload and return an anomaly detection result.

    The image is scored against the OK memory bank for the specified part type.
    A raw distance above the calibrated threshold is classified as NOT_OK.

    Query parameters
    ----------------
    part_type : str, required
        Part type identifier. Examples: part_A, part_B, part_C.

    Response fields
    ---------------
    status        : "OK" or "NOT_OK"
    confidence    : confidence in the verdict (0-100 %)
    anomaly_score : normalised anomaly score (0.0-1.0)
    part_type     : echoed back for traceability
    """
    try:
        validated_part = validate_part_type(part_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")
        if len(image_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum upload size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            )

        result = run_prediction(image_bytes, part_type=validated_part)
        return PredictionResponse(**result)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to analyze image.") from exc
