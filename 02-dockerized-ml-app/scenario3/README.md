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

# Build images
cd ..
docker build -t db-app ./dbapp
docker build -t web-app ./webapp

# Run containers
docker run -d -p 8001:8001 --name db-container db-app
# Train model (on host) before running web (creates webapp/model.pkl)
python webapp/train_model.py
docker run -d -p 8000:8000 --name web-container --env DBAPP_URL=http://host.docker.internal:8001 web-app

# Open
# http://localhost:8000

# Stop and remove
# docker stop web-container db-container
# docker rm web-container db-container
```


