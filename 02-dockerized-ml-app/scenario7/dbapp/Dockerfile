# ---------- Stage 1: Build Dependencies ----------
FROM python:3.13-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

# ---------- Stage 2: Final Image ----------
FROM python:3.13-slim
WORKDIR /app
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache-dir /wheels/*
COPY . .
EXPOSE 8001
CMD ["python", "app.py"]


