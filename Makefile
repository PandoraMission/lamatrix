.PHONY: all clean pytest coverage flake8 black mypy isort

# Run all the checks which do not change files
all: isort black flake8 pytest

# Run the unit tests using `pytest`
pytest:
	poetry run pytest src tests

# Lint the code using `flake8`
flake8:
	poetry run flake8 src tests

# Automatically format the code using `black`
black:
	poetry run black src tests

# Order the imports using `isort`
isort:
	poetry run isort src tests

# Release a version 
release:
	git commit -m "Bump version to $$(poetry version -s)"
	git tag -a "v$$(poetry version -s)" -m "Release v$$(poetry version -s)"
	git push origin main --tags

builddocs:
	poetry run sphinx-build -M html docs _build

servedocs: 
	poetry run python -m http.server --directory _build/html 8000