from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import os, sys
sys.path.append("/opt/airflow")  # если нужно

def train_register():
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://mlflow:5001")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "minio")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minio123")
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
    from training.train_v1_tfidf import main as train_main
    train_main(sample_train=8000, sample_test=3000)

with DAG(
    "imdb_train_register",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args={"retries": 1}
) as dag:
    t1 = PythonOperator(task_id="train_register_v1", python_callable=train_register)
    t1