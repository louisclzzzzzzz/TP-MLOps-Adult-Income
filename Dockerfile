FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and data
COPY src/ src/
COPY data/ data/

# Train the model and generate artifacts
# This embeds a production-ready model into the image
RUN PYTHONPATH=src python -m mlops_tp.train

# Data no longer needed at runtime — src/mlops_tp/artifacts/ holds the model
EXPOSE 8000

ENV PYTHONPATH=src

# PORT can be overridden by the hosting platform (e.g. Render sets $PORT automatically)
CMD sh -c "uvicorn mlops_tp.api:app --host 0.0.0.0 --port ${PORT:-8000}"
