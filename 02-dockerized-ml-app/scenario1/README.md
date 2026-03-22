```
# Initialize and manage deps with uv (optional)
uv init --python 3.13.5
uv add flask requests scikit-learn numpy
uv pip freeze > requirements.txt

# Train model (creates model.pkl)
python train_model.py

# Build and run Docker
docker build -t flask-ml-app .
docker run -p 5000:5000 flask-ml-app

# Stop container
# docker ps
# docker stop <id-or-name>
```


