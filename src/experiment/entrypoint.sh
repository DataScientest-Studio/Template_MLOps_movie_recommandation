#!/bin/bash

echo "➡️  Lancement du serveur MLflow..."
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri /app/mlruns \
  --default-artifact-root /app/mlruns &

# Attente pour laisser le temps au serveur de démarrer
sleep 5

echo "➡️  Lancement de l'expérience MLflow..."
#python src/experiment/grid_search_experiment.py
mlflow run src/experiment \
  --env-manager=local \
  --experiment-name=Movie_Recommandation_Model

echo "✅ Expérience terminée. MLflow disponible sur http://localhost:5000"
tail -f /dev/null