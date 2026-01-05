FROM python:3.10-slim

# Outils de compilation pour scikit-surprise
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dépendances Python
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# Code de l'application
COPY . .

# Cloud Run
ENV PORT=8080
CMD ["gunicorn", "-b", ":8080", "main:app"]
