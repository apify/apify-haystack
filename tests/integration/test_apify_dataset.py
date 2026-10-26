from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from haystack import Document

from apify_haystack import ApifyDatasetLoader

if TYPE_CHECKING:
    from collections.abc import Iterator

    from apify_client import ApifyClient

ITEMS = [
    {'url': 'https://example.com/1', 'markdown': '# First', 'metadata': {'title': 'First'}},
    {'url': 'https://example.com/2', 'markdown': '# Second', 'metadata': {'title': 'Second'}},
]


def to_document(item: dict) -> Document:
    return Document(content=item['markdown'], meta={'url': item['url']})


@pytest.fixture
def dataset_id(apify_client: ApifyClient) -> Iterator[str]:
    """An unnamed dataset with `ITEMS`, deleted after the test."""
    dataset = apify_client.datasets().get_or_create()
    apify_client.dataset(dataset.id).push_items(ITEMS)
    yield dataset.id
    apify_client.dataset(dataset.id).delete()


def test_loader_loads_dataset(dataset_id: str) -> None:
    """The loader reads every item of an existing dataset through the mapping function."""
    result = ApifyDatasetLoader(dataset_id, to_document).run()

    assert result == {'documents': [to_document(item) for item in ITEMS]}


def test_loader_uses_default_mapping_function(dataset_id: str) -> None:
    """Without a mapping function, items become Markdown documents with the page details in `meta`."""
    result = ApifyDatasetLoader(dataset_id).run()

    assert result == {
        'documents': [
            Document(content='# First', meta={'url': 'https://example.com/1', 'title': 'First'}),
            Document(content='# Second', meta={'url': 'https://example.com/2', 'title': 'Second'}),
        ]
    }


async def test_loader_run_async_loads_dataset(dataset_id: str) -> None:
    """`run_async` reads the same documents through the async client."""
    loader = ApifyDatasetLoader(dataset_id, to_document)

    result = await loader.run_async()
    await loader.close_async()

    assert result == {'documents': [to_document(item) for item in ITEMS]}
