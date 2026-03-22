```
# Optional: set up deps with uv per service
cd dbapp
uv init --python 3.13.5
uv add flask flask_sqlalchemy SQLAlchemy
uv pip freeze > requirements.txt
cd ../webapp
uv init --python 3.13.5
uv add flask requests scikit-learn numpy
uv pip freeze > requirements.txt

# Build and run with Docker Compose
cd ..
python webapp/train_model.py
docker compose up --build -d

# Open
# http://localhost:8000

# Stop
# docker compose down
```


