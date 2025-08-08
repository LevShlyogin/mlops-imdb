import os, mlflow
from mlflow.tracking import MlflowClient

def load_model_from_registry(name: str, stage: str):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    client = MlflowClient()
    mv = client.get_latest_versions(name=name, stages=[stage])[0]
    uri = f"models:/{name}/{stage}"
    model = mlflow.pyfunc.load_model(uri)
    return model, mv.version