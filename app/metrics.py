from prometheus_client import Counter, Histogram, Gauge

REQUESTS = Counter("requests_total", "Total requests", ["endpoint", "http_status", "model_version"])
LATENCY = Histogram("request_latency_seconds", "Request latency", ["endpoint", "model_version"])
MODEL_INFO = Gauge("model_info", "Model info as labels", ["name", "stage", "version"])