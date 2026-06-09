"""Parts API route — returns list of trained part types available for inference."""

from __future__ import annotations

from fastapi import APIRouter

from model.efficientnet import get_available_part_types

router = APIRouter(tags=["parts"])


@router.get("/parts", summary="List available trained part types")
async def list_parts() -> dict:
    """
    Return the list of part types that have a trained model available.

    Scans models/ directory for subdirectories containing anomaly_model.pkl.
    The frontend uses this endpoint to populate the Part Type selector dropdown.

    Response
    --------
    {
        "parts": ["part_A", "part_B", "part_C"]
    }

    An empty list means no parts have been trained yet.
    """
    available = get_available_part_types()
    return {"parts": available}
