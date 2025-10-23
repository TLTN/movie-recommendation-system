# Movie Recommendation System Makefile

.PHONY: help setup install download explore train test web clean all

# Default target
help:
	@echo "Movie Recommendation System"
	@echo "Available commands:"
	@echo "  setup      - Setup project directories"
	@echo "  install    - Install dependencies"
	@echo "  download   - Download MovieLens dataset"
	@echo "  explore    - Run data exploration"
	@echo "  train      - Train all models"
	@echo "  test       - Run tests"
	@echo "  web        - Start web application"
	@echo "  clean      - Clean generated files"
	@echo "  all        - Run complete pipeline"

# Setup project
setup:
	@echo "Setting up project..."
	python run_system.py --setup

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

# Download data
download:
	@echo "Downloading MovieLens dataset..."
	python run_system.py --download

# Data exploration
explore:
	@echo "Running data exploration..."
	python run_system.py --explore

# Train models
train:
	@echo "Training models..."
	python run_system.py --train

# Train specific models
train-knn:
	@echo "Training KNN model..."
	python experiments/train_knn.py --k 40 --similarity cosine --based user

train-mf:
	@echo "Training Matrix Factorization model..."
	python experiments/train_matrix_factorization.py --n-factors 50 --n-epochs 50

# Run tests
test:
	@echo "Running tests..."
	python run_system.py --test

# Start web application
web:
	@echo "Starting web application..."
	python src/web/app.py

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

# Clean all data and models
clean-all: clean
	@echo "Cleaning data and models..."
	rm -rf data/processed/
	rm -rf saved_models/*.pkl
	rm -rf results/
	rm -rf logs/

# Run complete pipeline
all:
	@echo "Running complete pipeline..."
	python run_system.py --all

# Development setup
dev-setup: install setup download
	@echo "Development environment ready!"

# Quick start for demo
demo: setup install download train web
	@echo "Demo ready! Access http://localhost:5000"

# Lint code (if you have flake8 installed)
lint:
	@echo "Linting code..."
	flake8 src/ tests/ --max-line-length=100

# Format code (if you have black installed)
format:
	@echo "Formatting code..."
	black src/ tests/ experiments/ --line-length=100

# Install development dependencies
install-dev:
	@echo "Installing development dependencies..."
	pip install -r requirements.txt
	pip install pytest flake8 black jupyter

# Run Jupyter notebook
notebook:
	@echo "Starting Jupyter notebook..."
	jupyter notebook notebooks/

# Check system requirements
check:
	@echo "Checking system requirements..."
	python -c "import sys; print(f'Python version: {sys.version}')"
	python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
	python -c "import pandas; print(f'Pandas version: {pandas.__version__}')"
	python -c "import sklearn; print(f'Scikit-learn version: {sklearn.__version__}')"

# Show project structure
tree:
	@echo "Project structure:"
	@tree -I '__pycache__|*.pyc|.git|venv|env' -a