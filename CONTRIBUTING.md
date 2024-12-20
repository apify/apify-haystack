# Development

Here you'll find a contributing guide to get started with development.

## Environment

For local development, it is required to have Python 3.9 (or a later version) installed.

We use [Poetry](https://python-poetry.org/) for project management. Install it and set up your IDE accordingly.

## Dependencies

To install this package and its development dependencies, run:

```sh
make install-dev
```

## Code checking

To execute all code checking tools together, run:

```sh
make check-code
```

### Linting

We utilize [ruff](https://docs.astral.sh/ruff/) for linting, which analyzes code for potential issues and enforces consistent style. Refer to `pyproject.toml` for configuration details.

To run linting:

```sh
make lint
```

### Formatting

Our automated code formatting also leverages [ruff](https://docs.astral.sh/ruff/), ensuring uniform style and addressing fixable linting issues. Configuration specifics are outlined in `pyproject.toml`.

To run formatting:

```sh
make format
```

### Type checking

Type checking is handled by [mypy](https://mypy.readthedocs.io/), verifying code against type annotations. Configuration settings can be found in `pyproject.toml`.

To run type checking:

```sh
make type-check
```

## Documentation

We adhere to the [Google docstring format](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for documenting our codebase. Every user-facing class or method is documented. Documentation standards are enforced using [Ruff](https://docs.astral.sh/ruff/).

## Release process

Publishing new versions to [PyPI](https://pypi.org/project/crawlee) is automated through GitHub Actions.

- **Beta releases**: On each commit to the master branch, a new beta release is automatically published. The version number is determined based on the latest release and conventional commits. The beta version suffix is incremented by 1 from the last beta release on PyPI.
- **Stable releases**: A stable version release may be created by triggering the `run_release` GitHub Actions workflow. The version number is determined based on the latest release and conventional commits (`auto` release type), or it may be overriden using the `custom` release type.

Important notes:

- Ensure the version number in `pyproject.toml` is updated before creating a new release. If a stable version with the same version number already exists on PyPI, the publish process will fail.
- The release process also fails if the version is not documented in `CHANGELOG.md`. Make sure to describe the changes in the new version there.
- After a stable release, ensure to increment the version number in both `pyproject.toml`.

### Beta release checklist

Before merging a pull request or committing directly to master for automatic beta release:

- Ensure [`pyproject.toml`](pyproject.toml) version reflects the latest non-published version.
- Describe changes in [`CHANGELOG.md`](CHANGELOG.md) under the latest non-published version section.

### Production release checklist

Before creating a GitHub Release for production:

- Confirm successful deployment of the latest beta release with the latest commit.
- Ensure changes are documented in `CHANGELOG.md` since the last production release.
- When drafting a GitHub release:
    - Create a new tag like `v1.2.3` targeting the master branch.
    - Use `1.2.3` as the release title.
    - Copy changes from `CHANGELOG.md` into the release description.
    - Check "Set as the latest release" option for visibility.

### Manual releases

To release a new version manually, follow these steps. Note that manual releases should only be performed if you have a good reason, use the automated release process otherwise.

1. Update the version number:

- Modify the `version` field under `tool.poetry` in `pyproject.toml`.
- Update the `__version__` field in `apify_haystack/__init__.py`.

```toml
[tool.poetry]
name = "apify-haystack"
version = "x.z.y"
```

2. Generate the distribution archives for the package:

```shell
poetry build
```

3. Set up the PyPI API token for authentication:

```shell
poetry config pypi-token.pypi YOUR_API_TOKEN
```

4. Upload the package to PyPI:

```shell
poetry publish
```
