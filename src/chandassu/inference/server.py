"""Local paste interface and JSON API, with one model loaded at startup."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .collection import ModelCollection


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=16384)
    strategy: str | None = None


def create_app(model_dir, device="cpu", cpu_threads=4, strategy="single"):
    folders = list(model_dir) if isinstance(model_dir, (list, tuple)) else [model_dir]
    predictor = ModelCollection(folders, device=device, cpu_threads=cpu_threads, strategy=strategy)
    app = FastAPI(title="Chandassu inference", version="1.0.0")
    app.state.predictor = predictor

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (Path(__file__).parents[1] / "web/index.html").read_text()

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model_type": predictor.members[0].config["model_type"],
            "loaded_models": len(predictor.members),
        }

    @app.get("/api/info")
    def info():
        return predictor.info()

    @app.post("/api/predict")
    def predict(request: PredictionRequest):
        try:
            return predictor.predict(request.text, request.strategy)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app
