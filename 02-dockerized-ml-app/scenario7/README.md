```
# Build multi-stage images
docker build -t dbapp_multistage ./dbapp
docker build -t webapp_multistage ./webapp

# Run containers
docker run -d -p 8001:8001 --name cntdbapp dbapp_multistage
python webapp/train_model.py
docker run -d -p 8000:8000 --name cntwebapp --env DBAPP_URL=http://host.docker.internal:8001 webapp_multistage

# Open
# http://localhost:8000

# Or use Compose
python webapp/train_model.py
docker compose up --build -d

# Stop
# docker stop cntwebapp cntdbapp
# docker rm cntwebapp cntdbapp
# docker compose down
```


