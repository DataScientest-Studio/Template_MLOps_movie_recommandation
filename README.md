# Reco Film

This project aims to provide a service to recommend movies to users. It is developed as the final project of MLOps formation from DataScientest.

## Run the project

This project uses a [docker-compose](docker-compose.yml) to provides `services` relatives to our projects.
Here is the list of all our services:
* `api` provides project API 
* `dev` to develop the project using a docker container

First build docker images:
```
docker-compose build
```

### Run API

**NB:** You can run API with the credentials: `username`: 'alice', `password`: 'x'.

* Run API:
```
docker compose up api
```

When the API is running, the API documentation is provided here: [http://172.19.0.5:8000/docs](http://172.19.0.5:8000/docs).

* Test API:
```
curl -X GET -i http://172.19.0.5:8000/
```

* Infer using model into API:
```
curl -X 'GET' \
  'http://localhost:8000/recommend/1' \
  -H 'accept: application/json'
```

* Kill API:
```
docker container stop reco_api
```

### Dev mode

* Run the dev container:
```
docker compose run --rm dev
```

You are now inside the container. 

* Construct the dataset:
```
poetry run python app.py dataset
```

* Prediction:
```
poetry run python app.py predict
```

* Test
```
poetry run pytest -s
```

### Dev API
* Run API:
```
uvicorn api:api --host 0.0.0.0 --port 8000 --reload
```

* See doc here:[http://localhost:8000/docs](http://localhost:8000/docs)


## Technical documentation
* [Cahier des charges](https://docs.google.com/document/d/11B940u5Z8is7Wlj7b0wzkA4UE-wqW9eDQh4Fcw8SisY/edit?usp=sharing)

### 0. Dev choices

* To develop we recommend to use VSCode with extension [ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff).

* We choose to use [poetry](https://python-poetry.org/) instead of `requirement.txt` because it *Python packaging and dependency management made easy*.


### 1. Architecture
TODO
```mermaid
architecture-beta
    group api(cloud)[API Reco Film]

    service db(database)[Database] in api
    service disk1(disk)[Storage] in api
    service server(server)[Server] in api

    db:L -- R:server
    disk1:T -- B:server

```

### 2. Docker and Docker-compose
TODO

### 3. API
[API](api.py) is implemented using [FastAPI](https://fastapi.tiangolo.com/). It contains a `dict` object to record user names and passwords. Passwords are encoded using [md5 algorithm](https://www.geeksforgeeks.org/md5-hash-python/).


## Useful
* [project template](https://github.com/DataScientest-Studio/Template_MLOps_movie_recommandation)
* [project board](https://github.com/users/Chrisdml/projects/1)
