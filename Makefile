.PHONY: clean install-dev build publish-to-pypi lint type-check check-code format check-version-conflict check-changelog-entry check-code build-api-reference run-doc

DIRS_WITH_CODE = src


clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache build dist htmlcov .coverage

install-dev:
	python3 -m pip install --upgrade pip poetry
	poetry install --all-extras
	poetry run pre-commit install

build:
	poetry build --no-interaction -vv

# APIFY_PYPI_TOKEN_APIFY_HAYSTACK is expected to be set in the environment
publish-to-pypi:
	poetry config pypi-token.pypi "${APIFY_PYPI_TOKEN_APIFY_HAYSTACK}"
	poetry publish --no-interaction -vv

lint:
	poetry run ruff format --check $(DIRS_WITH_CODE)
	poetry run ruff check $(DIRS_WITH_CODE)

type-check:
	poetry run mypy $(DIRS_WITH_CODE)
	#poetry run pytest --verbose --cov=src/apify_haystack --cov-report=html

format:
	poetry run ruff check --fix $(DIRS_WITH_CODE)
	poetry run ruff format $(DIRS_WITH_CODE)

check-code: lint type-check
