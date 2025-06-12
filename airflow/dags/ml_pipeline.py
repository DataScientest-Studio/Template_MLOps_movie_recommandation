from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from datetime import datetime
import os

default_args = {
    'owner': 'airflow',
    'retries': 1,
}

HOST_PATH = os.getenv("HOST_PATH")

vol1 = os.path.join(HOST_PATH, 'data')
vol2 = os.path.join(HOST_PATH, 'models')
vol3 = os.path.join(HOST_PATH, 'metrics')
vol4 = os.path.join(HOST_PATH, 'airflow/dags')
vol5 = os.path.join(HOST_PATH, 'airflow/logs')
vol6 = os.path.join(HOST_PATH, 'airflow/plugins')

# Fonction pour créer un DockerOperator
def create_docker_task(task_id, image, command):
    return DockerOperator(
        task_id=task_id,
        image=image,
        command=command,
        auto_remove=True,
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        mounts=[
            Mount(source=vol1, target='/app/data', type='bind'),
            Mount(source=vol2, target='/app/models', type='bind'),
            Mount(source=vol3, target='/app/metrics', type='bind'),
            Mount(source=vol4, target='/opt/airflow/dags', type='bind'),
            Mount(source=vol5, target='/opt/airflow/logs', type='bind'),
            Mount(source=vol6, target='/opt/airflow/plugins', type='bind'),
            Mount(source='/var/run/docker.sock', target='/var/run/docker.sock', type='bind'),
        ],
        force_pull=False,
    )

# Définition du DAG
with DAG(
    'ml_pipeline',
    default_args=default_args,
    description='Pipeline ML orchestration with Docker',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:
    # Création des tâches
    import_data = create_docker_task(
        task_id='import_data',
        image='maquirog/import_raw_data:latest',
        command='python src/data/import_raw_data.py',
    )

    build_features = create_docker_task(
        task_id='build_features',
        image='maquirog/build_features:latest',
        command='python src/data/build_features.py',
    )

    train_model = create_docker_task(
        task_id='train_model',
        image='maquirog/train:latest',
        command='python src/models/train.py',
    )

    predict_model = create_docker_task(
        task_id='predict_model',
        image='maquirog/predict:latest',
        command='python src/models/predict.py',
    )

    evaluate_model = create_docker_task(
        task_id='evaluate_model',
        image='maquirog/evaluate:latest',
        command='python src/models/evaluate.py',
    )

    # Définition des dépendances
    import_data >> build_features >> train_model >> predict_model >> evaluate_model