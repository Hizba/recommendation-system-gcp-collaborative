FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

COPY main.py .

ENV PORT=8080
CMD ["gunicorn", "-b", ":8080", "main:app"]
