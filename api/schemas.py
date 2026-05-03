"""
api/schemas.py — Pydantic request & response models for the AirSight API.

All float values are rounded to 2 decimal places in the serialiser so that
JSON responses stay compact and human-readable.
"""

from __future__ import annotations
from pydantic import BaseModel, Field


# ── Shared constants ──────────────────────────────────────────────────────────

POLLUTANT_UNITS: dict[str, str] = {
    "PM2.5": "µg/m³",
    "PM10":  "µg/m³",
    "O3":    "ppb",
    "CO":    "ppm",
    "SO2":   "ppb",
    "NO2":   "ppb",
    "AQI":   "—",
}


# ── Response models ───────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """
    Returned by POST /predict.
    Contains only the CNN pollutant predictions — no LLM call.
    """

    aqi: float = Field(..., description="Predicted AQI (0–500+ scale)", example=186.36)
    category: str = Field(..., description="AQI health category", example="Unhealthy")

    predictions: dict[str, float] = Field(
        ...,
        description="Per-pollutant predicted values in real-world units",
        example={
            "PM2.5": 167.52, "PM10": 197.82,
            "O3": 68.11,    "CO":   37.40,
            "SO2": 8.67,    "NO2":  57.47,
            "AQI": 186.36,
        },
    )
    units: dict[str, str] = Field(
        default=POLLUTANT_UNITS,
        description="Unit for each predicted pollutant",
    )
    processing_time_ms: float = Field(
        ..., description="End-to-end latency in milliseconds", example=1842.0
    )

    model_config = {"json_schema_extra": {"title": "Prediction Response"}}


class AnalysisResponse(PredictionResponse):
    """
    Returned by POST /analyze.
    Extends PredictionResponse with a RAG-grounded health advisory.
    """

    advisory: str = Field(
        ...,
        description="Gemini-generated health advisory grounded in retrieved PDF context",
    )
    sources: list[str] = Field(
        ...,
        description="PDF filenames used as RAG context for the advisory",
        example=["About_AQI.pdf", "aqi-technical-assistance-document-sept2018.pdf"],
    )

    model_config = {"json_schema_extra": {"title": "Analysis Response"}}


class HealthResponse(BaseModel):
    """Returned by GET /health."""

    status: str = Field("ok", example="ok")
    model_loaded: bool = Field(..., example=True)
    vectorstore_loaded: bool = Field(..., example=True)
    version: str = Field("1.0.0", example="1.0.0")


class ChatRequest(BaseModel):
    """
    Body for POST /chat.
    Send a plain-text question about air quality.
    Optionally include the predictions dict from a previous /analyze call
    to give Gemini context about the specific situation.
    """

    question: str = Field(
        ...,
        description="Air quality question to answer",
        example="What are the long-term health effects of high PM2.5 exposure?",
    )
    predictions: dict[str, float] | None = Field(
        default=None,
        description="(Optional) Predictions from a previous /analyze call for context",
        example={"PM2.5": 167.52, "AQI": 186.36},
    )


class ChatResponse(BaseModel):
    """Returned by POST /chat."""

    answer: str = Field(..., description="Gemini answer grounded in retrieved PDF context")
    sources: list[str] = Field(
        ...,
        description="PDF filenames used as RAG context",
        example=["About_AQI.pdf"],
    )
    processing_time_ms: float = Field(..., example=3210.0)
