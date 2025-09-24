format:
	uv run isort . --profile black --skip data
	uv run black . --verbose

lint:
	uv run flake8 spatialproteomics tests --exclude .venv,data,build,dist

test:
	uv run python -m pytest tests -x

all: format lint test
