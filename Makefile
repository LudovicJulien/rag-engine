install:
	pip install -e ".[dev]"
	pre-commit install

lint:
	black --check .
	isort --check .
	flake8 .
	mypy src/ tests/

test:
	pytest

run:
	uvicorn src.api.main:app --reload