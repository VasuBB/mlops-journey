# mlops-journey

A hands-on progression through core MLOps practices — from serving a basic ML model to full CI/CD pipelines with monitoring.

## Structure

| Folder | Topic |
|--------|-------|
| `01-flask-ml-app` | Serving an ML model via a Flask API (multiple scenarios) |
| `02-dockerized-ml-app` | Containerizing the Flask ML app with Docker |
| `03-experiment-tracking-mlflow` | Tracking experiments and runs with MLflow (sklearn & PyTorch) |
| `04-cicd-jenkins` | Automating build/test/deploy with a Jenkins pipeline |
| `05-model-versioning` | Model versioning and registry with MLflow |
| `06-monitoring-prometheus` | Exposing app metrics and alerting with Prometheus & Alertmanager |

## Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Jenkins (for `04-cicd-jenkins`)
- Prometheus & Alertmanager (for `06-monitoring-prometheus`)

## Quick Start

Each folder is self-contained. Refer to the `README.md` inside each folder for setup and run instructions.
