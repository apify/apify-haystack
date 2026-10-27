# Upgrading to 1.0

Version 1.0 has breaking changes from 0.1.x. This guide covers each one and what to change in your code. For the full list of changes, see the [changelog](https://github.com/apify/apify-haystack/blob/main/CHANGELOG.md).

## Python 3.11 or newer

Support for Python 3.9 and 3.10 is dropped. The package supports Python 3.11 to 3.15.

## Haystack 3 and apify-client 3

The package requires `haystack-ai>=3.0.0` and `apify-client>=3.4.0`. Haystack 3 has breaking changes of its own, so migrate the rest of your pipelines with it. If your code calls `apify-client` directly, update it for the 3.x API as well.

Haystack 3 only deserializes classes and functions from trusted modules. To load a saved pipeline, pass `allowed_modules=['apify_haystack']` to `Pipeline.loads`, along with the module of your mapping function if you use a custom one.

## The token parameter is `apify_token` and takes a `Secret`

`ApifyDatasetFromActorCall` and `ApifyDatasetFromTaskCall` take the token as `apify_token`, a Haystack `Secret`. The `apify_api_token` string parameter is gone.

```python
# 0.1.x
actor = ApifyDatasetFromActorCall('apify/website-content-crawler', mapping, apify_api_token='MY-TOKEN')

# 1.0
from haystack.utils import Secret

actor = ApifyDatasetFromActorCall(
    'apify/website-content-crawler', mapping, apify_token=Secret.from_env_var('MY_APIFY_TOKEN')
)
```

If you set the token as an environment variable, you don't need to change anything. The components read `APIFY_TOKEN` first and fall back to `APIFY_API_TOKEN`. A missing token raises a `ValueError` when the component creates its client, on `warm_up` or the first `run`, not in `__init__`.

`ApifyDatasetLoader` also takes `apify_token` now. Without a token, it can still load public datasets.

## The client is created in `warm_up`

The components follow the Haystack 3 lifecycle. Each one creates its Apify client in `warm_up`, or on the first `run`, and releases it in `close`. Before that, the `client` attribute is `None`. If your code used `component.client` right after creating the component, call `component.warm_up()` first.

## Failed runs raise an error

When an Actor or task run finishes with any status other than `SUCCEEDED`, `ApifyDatasetFromActorCall` and `ApifyDatasetFromTaskCall` raise a `RuntimeError`. In 0.1.x, they loaded whatever the run had written to its dataset. The error message includes the run ID and its status.

## Import from the package root

All modules are private. Import the public API from `apify_haystack`:

```python
from apify_haystack import ApifyDatasetFromActorCall, ApifyDatasetFromTaskCall, ApifyDatasetLoader
```

The `apify_haystack.apify_dataset` and `apify_haystack.constants` modules are gone, and so is `apify_haystack.examples`. The examples are in the [examples directory](https://github.com/apify/apify-haystack/tree/main/examples) of the repository.

## python-dotenv is not installed

The package no longer depends on `python-dotenv`. If your code loads a `.env` file with it, add `python-dotenv` to your own dependencies.
