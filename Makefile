.PHONY: help install extract train evaluate test lint app api docker-build docker-up

help:
	@echo "Available commands:"
	@echo "  make install     - Install python dependencies"
	@echo "  make extract     - Pre-extract and cache CNN spatial features"
	@echo "  make train       - Train Image Caption Generator"
	@echo "  make evaluate    - Run quantitative benchmark evaluation"
	@echo "  make test        - Run test suite via pytest"
	@echo "  make app         - Launch interactive Streamlit dashboard"
	@echo "  make api         - Launch FastAPI REST API server"
	@echo "  make docker-up   - Run services using Docker Compose"

install:
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('wordnet')"

extract:
	python scripts/extract_features.py --backbone resnet50 --batch-size 32

train:
	python scripts/train.py --epochs 15 --batch-size 32 --backbone resnet50

evaluate:
	python scripts/evaluate.py --method beam --beam-width 5

test:
	pytest -v tests/

app:
	streamlit run app/streamlit_app.py

api:
	uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-up:
	docker-compose -f docker/docker-compose.yml up
