"""
API response schemas for the anomaly detection inspection system.

Schema is intentionally extensible:
  - Phase 1: status + confidence + anomaly_score (active)
  - Phase 2: defect_type + bounding_boxes (YOLO integration, optional)
  - Phase 3: heatmap (Grad-CAM explainability, optional)

Adding Phase 2/3 fields to the API response requires no changes to
existing consumers — all new fields carry sensible None/[] defaults.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """
    Unified inspection result returned by POST /predict.

    Core fields (Phase 1 — always populated)
    -----------------------------------------
    status        : Overall inspection verdict.
    confidence    : Model confidence in the status verdict (0–100 %).
    anomaly_score : Normalised anomaly score (0.0 = perfectly normal,
                    1.0 = maximally anomalous).

    Future fields (Phase 2+ — not populated until implementation)
    -------------------------------------------------------------
    defect_type   : Specific defect class name (YOLO integration).
    bounding_boxes: List of defect bounding box coordinates.
    heatmap       : Base64-encoded or URL to a Grad-CAM anomaly heatmap.
    """

    # ------------------------------------------------------------------ #
    # Phase 1 — Anomaly detection core                                    #
    # ------------------------------------------------------------------ #
    status: Literal["OK", "NOT_OK"]
    confidence: float = Field(..., ge=0.0, le=100.0, description="Confidence %")
    anomaly_score: float = Field(
        ..., ge=0.0, le=1.0, description="Normalised anomaly score [0, 1]"
    )

    # Diagnostics / monitoring fields
    raw_distance: float = Field(
        default=0.0,
        description="Raw L2 distance to nearest OK embedding in memory bank",
    )
    threshold: float = Field(
        default=0.0,
        description="Calibrated anomaly threshold (raw distance units)",
    )
    part_type: str | None = Field(
        default=None,
        description="Part type used for this prediction (e.g. 'part_A')",
    )

    # ------------------------------------------------------------------ #
    # Phase 2 — YOLO defect localisation (reserved, not yet implemented)  #
    # ------------------------------------------------------------------ #
    defect_type: str | None = Field(
        default=None, description="Defect class label (Phase 2 — YOLO)"
    )
    bounding_boxes: list[Any] = Field(
        default_factory=list,
        description="Defect bounding boxes (Phase 2 — YOLO)",
    )

    # ------------------------------------------------------------------ #
    # Phase 3 — Explainability (reserved, not yet implemented)            #
    # ------------------------------------------------------------------ #
    heatmap: str | None = Field(
        default=None, description="Anomaly heatmap URL or base64 (Phase 3 — Grad-CAM)"
    )

    model_config = {"json_schema_extra": {
        "example": {
            "status": "NOT_OK",
            "confidence": 87.5,
            "anomaly_score": 0.62,
            "defect_type": None,
            "bounding_boxes": [],
            "heatmap": None,
        }
    }}
