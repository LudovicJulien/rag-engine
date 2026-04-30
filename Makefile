install:
	pip install -e ".[dev]"
	pre-commit install

lint:
	black --check .
	isort --check .
	flake8 .
	mypy src/

test:
	# Exit code 5 = no tests collected, acceptable during early development
	pytest || [ $$? -eq 5 ]

run:
	uvicorn src.api.main:app --reload