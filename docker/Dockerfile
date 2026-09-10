# Use locally available Python image
FROM mustafahussien7/fifa-value-api:v1.0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Set working directory

# Copy and install minimal requirements
COPY docker/requirements.docker.txt requirements.txt
RUN pip install --no-cache-dir --default-timeout=600 --retries 10 torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --default-timeout=600 --retries 10 -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# Copy application source
COPY . .

EXPOSE 8000 8501

# Default launch command (FastAPI)
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
