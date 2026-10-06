"""
Iris classifier API.

Run:   uvicorn main:app --reload
UI:    http://127.0.0.1:8000/
Docs:  http://127.0.0.1:8000/docs
"""
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

# Resolve relative to this file, not the launch directory. Override with env var if needed.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "artifacts" / "iris_model.pkl"))
UI_PATH = BASE_DIR / "static" / "index.html"

# Populated at startup
state = {"bundle": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup, not per request.
    # Only unpickle files you created yourself: pickle can execute arbitrary code.
    try:
        with open(MODEL_PATH, "rb") as f:
            state["bundle"] = pickle.load(f)
    except Exception as e:  # keep the app up so /health can report the failure
        state["error"] = f"{type(e).__name__}: {e}"
    yield
    state["bundle"] = None


app = FastAPI(title="Iris Classifier API", version="1.0.0", lifespan=lifespan)


# ---------- Schemas ----------
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, lt=20, description="cm")
    sepal_width: float = Field(..., gt=0, lt=20, description="cm")
    petal_length: float = Field(..., gt=0, lt=20, description="cm")
    petal_width: float = Field(..., gt=0, lt=20, description="cm")

    model_config = {
        "json_schema_extra": {
            "example": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
        }
    }


class PredictionOut(BaseModel):
    species: str
    class_index: int
    probabilities: dict[str, float]


# ---------- Endpoints ----------
@app.get("/", include_in_schema=False)
def ui():
    """Simple browser UI that calls /health and /predict."""
    if not UI_PATH.exists():
        return HTMLResponse(
            "<h1>UI file missing</h1><p>Expected static/index.html next to main.py. "
            "The API still works: see <a href='/docs'>/docs</a>.</p>",
            status_code=404,
        )
    return FileResponse(UI_PATH)


@app.get("/health")
def health():
    bundle = state["bundle"]
    body = {"status": "ok" if bundle else "degraded", "model_loaded": bundle is not None}
    if bundle:
        body["model_name"] = bundle.get("model_name")
        body["trained_sklearn_version"] = bundle.get("sklearn_version")
        body["runtime_sklearn_version"] = sklearn.__version__
    else:
        body["error"] = state["error"]
    return body


@app.post("/predict", response_model=PredictionOut)
def predict(features: IrisFeatures):
    bundle = state["bundle"]
    if bundle is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {state['error']}")

    model = bundle["model"]
    names = bundle["feature_names"]
    labels = bundle["target_names"]

    # Build a 1-row DataFrame in the exact column order used during training
    row = pd.DataFrame([features.model_dump()])[names]

    idx = int(model.predict(row)[0])
    probs = model.predict_proba(row)[0] if hasattr(model, "predict_proba") else None

    return PredictionOut(
        species=labels[idx],
        class_index=idx,
        probabilities={l: round(float(p), 4) for l, p in zip(labels, probs)} if probs is not None else {},
    )