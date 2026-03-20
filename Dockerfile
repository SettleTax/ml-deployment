FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY settletax_classifier.py .
COPY api.py .

ENV PORT=8080

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
