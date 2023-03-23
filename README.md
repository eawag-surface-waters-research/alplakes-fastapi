# Alplakes FastAPI

This is a repository for Eawag SURF fastapi, initially developed for the ALPLAKES project.

Endpoints can be found in `app/main.py`

The documentation pages are available at the following:

- Swagger [http://localhost:8000/docs](http://localhost:8000/docs)
- Redoc [http://localhost:8000/redoc](http://localhost:8000/redoc)

## Production Deployment

### 1. Install docker

Follow the official install instructions [here](https://docs.docker.com/engine/install/)

Then run `docker login` to enable access to private docker images.

### 2. Clone repository
```console
sudo apt-get -y update && sudo apt-get -y install git
git clone https://github.com/eawag-surface-waters-research/alplakes-fastapi.git
mkdir -p filesystem
```

### 3. Launch service
```console
cd alplakes-fastapi
docker compose up --build -d
```

### 3. Restart service
When there are changes to the API the service needs to be rebuilt to reflect those changes.
```console
docker compose down
docker compose up --build -d
```

## Local Development

### 1. Install virtual environment

```console
conda create --name fastapi python=3.9.14
conda activate fastapi
conda install --file requirements.txt
```

### 2. Run API
```console
conda activate fastapi
uvicorn app.main:app --host 0.0.0.0 --reload
```

## Docker Commands

### Terminate containers
```console 
docker compose down
```

### List active containers
```console 
docker ps
```

### Go inside the container
```console 
docker exec -it 'container-id' bash
```
