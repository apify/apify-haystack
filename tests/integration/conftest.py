from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from apify_client import ApifyClient

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def apify_token() -> str:
    """The token of the Apify account the tests run on. The components read it from `APIFY_TOKEN` themselves."""
    token = os.environ.get('APIFY_TOKEN')
    if not token:
        pytest.skip('Integration tests need an Apify API token in the `APIFY_TOKEN` environment variable.')
    return token


@pytest.fixture
def apify_client(apify_token: str) -> Iterator[ApifyClient]:
    """A plain Apify client for preparing and cleaning up the resources a test uses."""
    client = ApifyClient(apify_token)
    yield client
    client.http_client.close()
