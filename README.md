# mlops-journey

A hands-on, end-to-end progression through core MLOps practices — from deploying a basic ML model with Flask to full CI/CD pipelines with production monitoring.

Each module builds on the previous one, introducing new tooling and concepts incrementally.

---

## Modules

### 01 — Flask ML App (`01-flask-ml-app`)
Serving a trained scikit-learn model via a Flask REST API with an HTML frontend. Explores multiple serving configurations across scenarios.

| Scenario | What it covers |
|----------|---------------|
| `scenario1` | Basic Flask app — train model, serve predictions |
| `scenario2` | Add a results page and improved UI |
| `scenario3–6` | Introduce Nginx as a reverse proxy in front of Flask |
| `final` | Combined reference config |

**Key concepts:** Flask routes, `pickle` model loading, Jinja2 templates, Nginx reverse proxy

---

### 02 — Dockerized ML App (`02-dockerized-ml-app`)
Containerizing the Flask ML app using Docker and Docker Compose, progressively splitting responsibilities across services.

| Scenario | What it covers |
|----------|---------------|
| `scenario1–2` | Single-container Flask app with Dockerfile |
| `scenario3–5` | Two-container setup: `webapp` + `dbapp` (separate Dockerfiles) |
| `scenario6–7` | Multi-service orchestration with `docker-compose.yml` |

**Key concepts:** Dockerfile, `.dockerignore`, Docker networking, multi-service Compose

---

### 03 — Experiment Tracking with MLflow (`03-experiment-tracking-mlflow`)
Logging and comparing ML training runs using MLflow on the Iris dataset.

| Sub-folder | What it covers |
|------------|---------------|
| `using-sklearn` | Log params, metrics, and artifacts with sklearn |
| `using-sklearn-hyperparameter` | Grid/random search with MLflow run comparison |
| `using-pytorch` | PyTorch training loop with MLflow tracking |

**Key concepts:** `mlflow.start_run`, `log_param`, `log_metric`, `log_artifact`, MLflow UI (`mlflow ui`)

---

### 04 — CI/CD with Jenkins (`04-cicd-jenkins`)
Automating build, test, and deployment of the Dockerized ML app using a declarative Jenkins pipeline.

**Structure:**
```
04-cicd-jenkins/
├── Jenkinsfile
├── docker-compose.yml
├── webapp/          # Wine prediction Flask app
└── dbapp/           # Record storage service
```

**Pipeline stages:** Checkout → Build Docker Images → Run Containers → Health Check

**Key concepts:** Declarative `Jenkinsfile`, Docker-in-Jenkins, pipeline triggers, health check validation

---

### 05 — Model Versioning with MLflow (`05-model-versioning`)
Registering, versioning, and promoting ML models using the MLflow Model Registry.

**Key concepts:** `mlflow.register_model`, model stages (Staging → Production), loading a versioned model for inference

---

### 06 — Monitoring with Prometheus & Alertmanager (`06-monitoring-prometheus`)
Instrumenting the Flask app with Prometheus metrics and configuring alert rules.

**Structure:**
```
06-monitoring-prometheus/
├── prometheus.yml       # Scrape config
├── alertmanager.yml     # Alert routing
├── rules.yml            # Alert thresholds
├── requirements.txt
├── webapp/              # Flask app with /metrics endpoint
└── dbapp/               # DB service with /metrics endpoint
```

**Key concepts:** `prometheus_client` (Counter, Histogram), `/metrics` endpoint, scrape intervals, Alertmanager routing

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Flask | ML model serving |
| scikit-learn / PyTorch | Model training |
| Docker & Docker Compose | Containerization |
| Nginx | Reverse proxy |
| MLflow | Experiment tracking & model registry |
| Jenkins | CI/CD pipeline automation |
| Prometheus | Metrics collection & alerting |
| Alertmanager | Alert routing and notification |

---

## Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Jenkins (for `04-cicd-jenkins`)
- Prometheus & Alertmanager binaries (for `06-monitoring-prometheus`)

---

## Getting Started

Each module is self-contained. Navigate into the folder and follow its `README.md`:

```bash
cd 02-dockerized-ml-app/scenario4
cat README.md
```

To run any Python module locally without Docker:

```bash
python -m venv .venv
source .venv/Scripts/activate    # Windows
pip install -r requirements.txt
python webapp/train_model.py
python webapp/app.py
```
