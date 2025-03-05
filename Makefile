# Variables
VENV_NAME = venv
REQUIREMENTS = requirements.txt
PYTHON = python
PIP = pip
LINTER = flake8
FORMATTER = black
SECURITY = bandit
DOCKER_IMAGE = fastapi-mlflow-app

install: $(VENV_NAME)/bin/activate
	$(PIP) install -r $(REQUIREMENTS)

$(VENV_NAME)/bin/activate:
	python -m venv $(VENV_NAME)	
	$(PIP) install --upgrade pip
	touch $(VENV_NAME)/bin/activate

format:
	$(FORMATTER) .

lint:
	$(LINTER) .

security:
	$(SECURITY) .

prepare:
	$(PYTHON) main.py --prepare

train:
	$(PYTHON) main.py --train
	
evaluate:
	$(PYTHON) main.py --evaluate

run-api:
	$(VENV_NAME)/bin/uvicorn app:app --reload

run-mlflow:
	$(VENV_NAME)/bin/mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

build-docker:
	docker build -t firas-aguech-4ds5-mlops .

run-docker:
	docker run -p 8000:8000 firas-aguech-4ds5-mlops


all: install format lint security prepare train evaluate
