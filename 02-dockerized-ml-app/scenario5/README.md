```
# Build images
docker build -t dbapp_image ./dbapp
docker build -t webapp_image ./webapp

# Create network and volume
docker network create appnet
docker volume create myvolume

# Train model (on host) before running web)
python webapp/train_model.py

# Run containers
docker run -p 8001:8001 -d --name dbapp --hostname dbapp -v myvolume:/dbapp/ --network appnet dbapp_image
docker run -p 8000:8000 -d --name cntwebapp --network appnet webapp_image

# Open
# http://localhost:8000

# Stop and remove
# docker stop cntwebapp dbapp
# docker rm cntwebapp dbapp
# docker network rm appnet
# docker volume rm myvolume
```


