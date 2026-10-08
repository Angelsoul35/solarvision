"""FastAPI surface over the SolarVision service."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from .detector import Assessment
from .labels import DEFECT_CLASSES, DefectClass
from .service import SolarVisionService

app = FastAPI(title="SolarVision", version="1.0.0")
_service: SolarVisionService | None = None


def service() -> SolarVisionService:
    global _service
    if _service is None:
        _service = SolarVisionService()
    return _service


class SampleRef(BaseModel):
    defect: str = Field(description="one of: healthy, hotspot, crack, dead_cell")
    seed: int = 0
    severity: float = Field(default=0.7, ge=0.0, le=1.0)
    size_frac: float = Field(default=0.6, ge=0.0, le=1.0)


class AssessRequest(BaseModel):
    """Provide exactly one of ``image_base64`` (a grayscale PNG) or ``sample``."""

    image_base64: str | None = None
    sample: SampleRef | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> AssessRequest:
        if (self.image_base64 is None) == (self.sample is None):
            raise ValueError("provide exactly one of 'image_base64' or 'sample'")
        return self


def _assessment_dict(a: Assessment) -> dict:
    return {
        "predicted_class": a.predicted_class,
        "title": a.title,
        "confidence": round(a.confidence, 4),
        "needs_review": a.needs_review,
        "decided_by": a.decided_by,
        "reason": a.reason,
        "evidence": a.evidence,
        "probabilities": a.probabilities,
    }


@app.post("/assess")
def assess(req: AssessRequest) -> dict:
    svc = service()
    if req.sample is not None:
        if req.sample.defect not in {c.value for c in DefectClass}:
            raise HTTPException(status_code=422, detail=f"unknown defect {req.sample.defect!r}")
        result = svc.assess_sample(
            req.sample.defect,
            seed=req.sample.seed,
            severity=req.sample.severity,
            size_frac=req.sample.size_frac,
        )
        return {
            "ground_truth": result.ground_truth,
            "correct": result.correct,
            "image_base64": result.image_base64,
            "assessment": _assessment_dict(result.assessment),
        }
    try:
        assessment = svc.assess_base64_png(req.image_base64 or "")
    except Exception as exc:  # malformed base64 or not an image
        raise HTTPException(status_code=422, detail=f"could not decode image: {exc}") from exc
    return {"assessment": _assessment_dict(assessment)}


@app.get("/classes")
def classes() -> dict:
    return {"classes": [c.value for c in DefectClass], "defects": DEFECT_CLASSES}
