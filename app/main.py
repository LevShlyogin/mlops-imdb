import time, os, pandas as pd
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
from app.schemas import PredictRequest, PredictResponse
from app.metrics import REQUESTS, LATENCY, MODEL_INFO
from app.model_loader import load_model_from_registry
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(title="IMDB Sentiment Service")

MODEL_NAME = os.getenv("MODEL_NAME", "sentiment-imdb")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Staging")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

model = None
model_version = "unknown"

@app.on_event("startup")
def startup_event():
    global model, model_version
    try:
        model, mv = load_model_from_registry(MODEL_NAME, MODEL_STAGE)
        model_version = str(mv)
        MODEL_INFO.labels(name=MODEL_NAME, stage=MODEL_STAGE, version=model_version).set(1)
    except Exception as e:
        print(f"Model loading failed: {e}")

@app.get("/healthz")
def healthz():
    status = "ok" if model is not None else "model_not_loaded"
    return {"status": status, "model_version": model_version}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, response: Response):
    start = time.time()
    status_code = 200
    try:
        preds = model.predict(pd.DataFrame({"text": [req.text]}))
        label = int(preds[0])
        return PredictResponse(label=label)
    except Exception:
        status_code = 500
        response.status_code = 500
        return PredictResponse(label=-1)
    finally:
        LATENCY.labels(endpoint="/predict", model_version=model_version).observe(time.time()-start)
        REQUESTS.labels(endpoint="/predict", http_status=str(status_code), model_version=model_version).inc()

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)