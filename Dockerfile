# 1. Utiliser une image légère Python
FROM python:3.10-slim

# 2. Définir le répertoire de travail
WORKDIR /app

# 3. Copier les fichiers du projet
COPY . .

# 4. Installer les dépendances
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 5. Exposer le port
EXPOSE 8000

# 6. Lancer l'application
CMD ["uvicorn", "api_fast.main:app", "--host", "0.0.0.0", "--port", "8000"]
