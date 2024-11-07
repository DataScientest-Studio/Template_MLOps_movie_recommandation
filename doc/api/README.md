# API

This document provides an exhaustive documentation about the API.

## Run the API
* In a terminal:
```
docker compose up api
```

* See API Documentation:
    * `http://172.19.0.5:8000/docs`
    * `http://172.19.0.5:8000/redoc`

* Test API is running:
``` 
curl -X GET -i http://172.19.0.5:8000/
```

* Kill API:
```
docker container stop reco_api
docker container rm reco_api
```

## Technical documentation
Our API call are implemented in file []() using the python framework FastAPI.

### Security