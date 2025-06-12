# Utiliser une image officielle Python comme base
FROM python:3.10-slim

# Définir le répertoire de travail à l'intérieur du conteneur
WORKDIR /app

# Copier les fichiers nécessaires pour tous les services
COPY ./src /app/src
COPY ./data /app/data
COPY ./evidently /app/evidently
COPY ./models /app/models
COPY ./metrics /app/metrics

COPY ./requirements/requirements.txt /app/requirements.txt

# Installer les dépendances
RUN pip install --no-cache-dir -r /app/requirements.txt

CMD ["python", "main.py"]