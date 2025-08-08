mlops-imdb/
├─ app/
│  ├─ main.py                 # FastAPI сервис (/predict, /healthz, /metrics)
│  ├─ model_loader.py         # Загрузка модели (MLflow URI или локальный путь)
│  ├─ schemas.py              # Pydantic схемы
│  ├─ metrics.py              # Prometheus метрики
│  └─ requirements.v1.txt     # fastapi, mlflow, sklearn, prometheus-client…
├─ training/
│  ├─ train_v1_tfidf.py       # Бейзлайн: TF-IDF + LogisticRegression → MLflow + Registry
│  └─ train_v2_hf.py          # (опц.) Регистрация модели на HF DistilBERT как pyfunc
├─ airflow/
│  └─ dags/
│     └─ train_register_imdb.py  # DAG: train→eval→register (MLflow)
├─ k8s/
│  ├─ namespace.yaml
│  ├─ deployment-v1.yaml
│  ├─ deployment-v2.yaml
│  ├─ service.yaml
│  └─ ingress.yaml
├─ docker/
│  ├─ Dockerfile.v1           # лёгкий образ со sklearn-моделью
│  └─ Dockerfile.v2           # (опц.) образ с transformers/torch
├─ compose/
│  ├─ docker-compose.yml      # MLflow+MinIO+Postgres+Prom+Grafana+Airflow
│  ├─ prometheus.yml
│  └─ grafana-datasource.yml
├─ .gitlab-ci.yml
├─ .env.example               # переменные для MinIO/MLflow
└─ README.md
