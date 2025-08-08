import os, mlflow, pandas as pd
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from mlflow.models.signature import infer_signature
import mlflow.sklearn
from mlflow.tracking import MlflowClient

EXPERIMENT = "imdb-sentiment"
MODEL_NAME = "sentiment-imdb"

def main(sample_train=10000, sample_test=5000):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    mlflow.set_experiment(EXPERIMENT)

    ds = load_dataset("imdb")
    train = ds["train"].shuffle(seed=42).select(range(sample_train))
    test = ds["test"].shuffle(seed=42).select(range(sample_test))

    X_train = [x["text"] for x in train]
    y_train = [int(x["label"]) for x in train]
    X_test = [x["text"] for x in test]
    y_test = [int(x["label"]) for x in test]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=1000, solver="liblinear"))
    ])

    with mlflow.start_run() as run:
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        mlflow.log_params({
            "model": "LogReg+TFIDF",
            "max_features": 50000,
            "ngram": "1-2",
            "solver": "liblinear",
            "sample_train": sample_train,
            "sample_test": sample_test
        })
        mlflow.log_metrics({"accuracy": acc, "f1": f1})

        X_sample = pd.DataFrame({"text": X_test[:5]})
        signature = infer_signature(X_sample, pipe.predict(X_sample))
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            signature=signature,
            input_example=X_sample,
            registered_model_name=MODEL_NAME
        )

        run_id = run.info.run_id

    client = MlflowClient()
    latest = client.get_latest_versions(name=MODEL_NAME, stages=[])
    # Переведём последнюю версию в Staging (демо-логика)
    mv = sorted(latest, key=lambda v: int(v.version))[-1]
    client.transition_model_version_stage(MODEL_NAME, mv.version, stage="Staging")
    print(f"Registered {MODEL_NAME} v{mv.version} to Staging; acc={acc:.4f} f1={f1:.4f}")

if __name__ == "__main__":
    main()