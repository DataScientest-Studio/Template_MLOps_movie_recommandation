import json  # For formatting JSON responses
from fastapi import FastAPI, APIRouter, HTTPException, Response
from fastapi.responses import JSONResponse  # For custom JSON responses
from src.api.models import TrainRequest, PredictionRequest
import docker
import os
from fastapi import Body

# Initialize FastAPI app
app = FastAPI()
router = APIRouter()

# Initialize Docker client
docker_client = docker.from_env(timeout=300)

def trigger_microservice(service_name: str, command: str = None):
    try:
        host_project_path = os.environ.get("HOST_PROJECT_PATH")
        # Decide how to mount volumes based on environment variable
        if os.environ.get("VOLUME_MODE", "bind") == "bind":
            # Use host paths for bind mounts
            volumes = {
                os.path.join(host_project_path, "src"): {"bind": "/app/src", "mode": "rw"},
                os.path.join(host_project_path, "data"): {"bind": "/app/data", "mode": "rw"},
                os.path.join(host_project_path, "models"): {"bind": "/app/models", "mode": "rw"},
                os.path.join(host_project_path, "metrics"): {"bind": "/app/metrics", "mode": "rw"},
                os.path.join(host_project_path, "reports"): {"bind": "/app/reports", "mode": "rw"},
            }
            # Debug printout
            print("Using bind mounts from host for volumes:")
            for host_path, mount_info in volumes.items():
                print(f"  Host: {host_path} -> Container: {mount_info['bind']} (mode: {mount_info['mode']})")
        else:
            # Use named Docker volumes (for CI/production)
            volumes = {
                "shared_src": {"bind": "/app/src", "mode": "rw"},
                "shared_data": {"bind": "/app/data", "mode": "rw"},
                "shared_models": {"bind": "/app/models", "mode": "rw"},
                "shared_metrics": {"bind": "/app/metrics", "mode": "rw"},
                "shared_reports": {"bind": "/app/reports", "mode": "rw"},
            }
            print("Using named Docker volumes for volumes:")
            for vol_name, mount_info in volumes.items():
                print(f"  Volume: {vol_name} -> Container: {mount_info['bind']} (mode: {mount_info['mode']})")

        # Create and run the container
        container = docker_client.containers.run(
            image=f"maquirog/{service_name}:latest",
            command=command,
            detach=True,
            volumes=volumes,
            working_dir="/app",
            stdout=True,
            stderr=True,
        )

        # Wait for the container to finish
        logs = container.logs(follow=True)
        decoded_logs = logs.decode("utf-8")
        print(f"🚀 Logs for {service_name} microservice:\n{decoded_logs}")

        # Remove the container after it stops
        container.remove()

        # Beautify the response JSON
        response = {
            "status": "success",
            "message": f"{service_name} microservice completed successfully",
            "logs": decoded_logs,
        }
        return JSONResponse(content=json.loads(json.dumps(response, indent=4)))
    except docker.errors.DockerException as e:
        print(f"Error while running container: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger {service_name}: {str(e)}")

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Microservice endpoints
@router.post("/import_raw_data")
def import_raw_data():
    return trigger_microservice("import_raw_data", command="python src/data/import_raw_data.py")

@router.post("/build_features")
def build_features():
    return trigger_microservice("build_features", command="python src/data/build_features.py")

#@router.post("/train")
#def train_model():
#    return trigger_microservice("train", command="python src/models/train.py")


# Training endpoint
@router.post("/train")
def train(request: TrainRequest = Body(default=None)):
    if not request or request.hyperparams is None:
        hyperparams_arg = ""
    else:
        json_params = json.dumps(request.hyperparams)
        hyperparams_arg = f"--hyperparams_dict '{json_params}'"

    command = f"python src/models/train.py {hyperparams_arg}"

    return trigger_microservice(
        service_name="train",
        command=command
    )

# Prediction endpoint
@router.post("/predict")
def predict(request: PredictionRequest = Body(default=None)):
    if not request:
        user_ids_arg = ""  # Default behavior: process all users
        n_recommendations = 10
    else:
        if request.user_ids:
            user_ids = ",".join(map(str, request.user_ids))
            user_ids_arg = f"--user_ids {user_ids}"
        else:
            user_ids_arg = ""  # Default behavior: all users
        n_recommendations = request.n_recommendations

    return trigger_microservice(
        service_name="predict",
        command=f"python src/models/predict.py {user_ids_arg} --n_recommendations {n_recommendations}"
    )

@router.post("/evaluate")
def evaluate():
    return trigger_microservice("evaluate", command="python src/models/evaluate.py")

@app.get("/prometheus_metrics", response_class=Response)
def get_prometheus_metrics():
    metrics_path = "metrics/scores.json"
    if not os.path.isfile(metrics_path):
        return Response(content="# Metrics file not found\n", media_type="text/plain", status_code=404)
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    prom_lines = []
    for key, value in metrics.items():
        metric_name = key.lower().replace("@", "_at_").replace(".", "_").replace("-", "_")
        prom_lines.append(f'{metric_name} {value}')
    prom_str = "\n".join(prom_lines) + "\n"
    return Response(content=prom_str, media_type="text/plain")

app.include_router(router)