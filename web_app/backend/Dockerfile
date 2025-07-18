# Utiliser une image officielle Python comme base
FROM python:3.10-slim

# Définir le dossier de travail
WORKDIR /app

# Copier les fichiers de l'application dans le conteneur
COPY . /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python
RUN pip install --upgrade pip
RUN pip install fastapi uvicorn[standard] torch torchvision \
    pillow opencv-python-headless ultralytics

# Exposer le port de l'application
EXPOSE 8080

# Lancer l'application FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
