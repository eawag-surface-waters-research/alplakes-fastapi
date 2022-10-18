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
docker-compose up --build -d
```

## Local Development

### 1. Install Pyenv
Install dependencies 
```console
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev
```
Download installer
```console
curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash
```
Check properly installed
```console
pyenv -v
```

### 2. Install virtual environment

```console
pyenv install 3.9.14
pyenv virtualenv 3.9.14 fastapi
pyenv activate fastapi
pip install -r requirements.txt
```

### 3. Run API
```console
pyenv activate fastapi
uvicorn app.main:app --host 0.0.0.0 --reload
```

## Docker Commands

### Terminate containers
```console 
docker-compose down
```

### List active containers
```console 
docker ps
```

### Go inside the container
```console 
docker exec -it 'container-id' bash
```
