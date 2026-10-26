# Development

Here you'll find a contributing guide to get started with development.

## Environment

For local development, you need a Python version that satisfies `requires-python` in `pyproject.toml`.

We use [uv](https://docs.astral.sh/uv/) for project management. Install it and set up your IDE accordingly.

We use [Poe the Poet](https://poethepoet.natn.io/) as a task runner, similar to npm scripts in `package.json`. All tasks are defined in `pyproject.toml` under `[tool.poe.tasks]`. Run `uv run poe` to list them with their descriptions, and `uv run poe <task>` to run one.

## Dependencies

To install this package with its development dependencies and the pre-commit hooks, run:

```sh
uv run poe install-dev
```

## Code checking

To run all code checks, run:

```sh
uv run poe check-code
```

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting and [mypy](https://mypy.readthedocs.io/) for type checking. Both are configured in `pyproject.toml`, and each has its own task (`lint`, `format`, `type-check`).

## Documentation

We adhere to the [Google docstring format](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for documenting our codebase. Every user-facing class or method is documented. Documentation standards are enforced using [Ruff](https://docs.astral.sh/ruff/).

## Release process

Publishing new versions to [PyPI](https://pypi.org/project/apify-haystack) is automated through GitHub Actions.

- **Beta releases**: On each commit to the main branch, a new beta release is automatically published. The version number is determined based on the latest release and conventional commits. The beta version suffix is incremented by 1 from the last beta release on PyPI.
- **Stable releases**: A stable version release may be created by triggering the `release` GitHub Actions workflow. The version number is determined based on the latest release and conventional commits (`auto` release type), or it may be overridden using the `custom` release type.

### Publishing to PyPI manually

1. **Do not do this unless absolutely necessary.** In all conceivable scenarios, you should use the `release` workflow instead.
2. **Make sure you know what you're doing.**
3. Update the version number by modifying the `version` field under `project` in `pyproject.toml`:

    ```toml
    [project]
    name = "apify_haystack"
    version = "x.z.y"
    ```

4. Build the package:

    ```sh
    uv run poe build
    ```

5. Upload to PyPI:

    ```sh
    uv publish --token YOUR_API_TOKEN
    ```
