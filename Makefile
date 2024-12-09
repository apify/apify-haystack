.PHONY: clean install-dev build publish-to-pypi lint type-check format check-version-conflict check-changelog-entry check-code

DIRS_WITH_CODE = src scripts

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
	poetry run ruff check $(DIRS_WITH_CODE)

type-check:
	poetry run mypy $(DIRS_WITH_CODE)

format:
	poetry run ruff check --fix $(DIRS_WITH_CODE)
	poetry run ruff format $(DIRS_WITH_CODE)

check-version-conflict:
	python3 scripts/check_version_conflict.py

check-changelog-entry:
	python3 scripts/check_changelog_entry.py

# The check-code target runs a series of checks equivalent to those performed by pre-commit hooks
# and the run_checks.yaml GitHub Actions workflow.
check-code: lint type-check check-version-conflict check-changelog-entry
