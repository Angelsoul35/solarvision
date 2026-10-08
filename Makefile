.PHONY: install test lint run demo eval clean

install:
	pip install -e ".[dev]"

test:
	pytest --cov=solarvision --cov-report=term-missing

lint:
	ruff check solarvision tests

run:
	uvicorn solarvision.main:app --reload --port 8000

demo:
	python -m solarvision.cli demo

eval:
	python -m solarvision.evaluation

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
