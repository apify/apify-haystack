from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import pytest
from haystack import Document

from apify_haystack import ApifyDatasetFromActorCall, ApifyDatasetFromTaskCall, ApifyDatasetLoader

if TYPE_CHECKING:
    from collections.abc import Iterator

    from apify_client import ApifyClient

ITEMS = [
    {'url': 'https://example.com/1', 'markdown': '# First', 'metadata': {'title': 'First'}},
    {'url': 'https://example.com/2', 'markdown': '# Second', 'metadata': {'title': 'Second'}},
]

# Cheerio Scraper is free, official, and fast. Its page function decides what lands in the dataset, and
# `example.com` keeps the page content fixed.
CHEERIO_SCRAPER = 'apify/cheerio-scraper'

PAGE_FUNCTION = """
async function pageFunction({ $, request }) {
    return { url: request.url, text: $('h1').text() };
}
"""

# The required `proxyConfiguration` is left out, so the platform fills in its default, Apify Proxy.
CHEERIO_INPUT = {
    'startUrls': [{'url': 'https://example.com'}],
    'pageFunction': PAGE_FUNCTION,
}

EXAMPLE_DOCUMENT = Document(content='Example Domain', meta={'url': 'https://example.com'})


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


@pytest.fixture
def task_id(apify_client: ApifyClient) -> Iterator[str]:
    """A Cheerio Scraper task with `CHEERIO_INPUT`, deleted after the test."""
    actor = apify_client.actor(CHEERIO_SCRAPER).get()
    assert actor is not None
    task = apify_client.tasks().create(
        actor_id=actor.id, name=f'apify-haystack-test-{uuid.uuid4().hex[:12]}', task_input=CHEERIO_INPUT
    )
    yield task.id
    apify_client.task(task.id).delete()


def test_actor_call_loads_run_dataset() -> None:
    """The component runs the Actor with the input merged from the constructor and `run`, and loads its dataset."""
    actor = ApifyDatasetFromActorCall(
        CHEERIO_SCRAPER,
        run_input={key: value for key, value in CHEERIO_INPUT.items() if key != 'startUrls'},
        memory_mbytes=1024,
    )

    result = actor.run(run_input={'startUrls': CHEERIO_INPUT['startUrls']})
    actor.close()

    assert result == {'documents': [EXAMPLE_DOCUMENT]}


async def test_actor_call_run_async_loads_run_dataset() -> None:
    """`run_async` runs the Actor through the async client and loads the same documents."""
    actor = ApifyDatasetFromActorCall(CHEERIO_SCRAPER, run_input=CHEERIO_INPUT, memory_mbytes=1024)

    result = await actor.run_async()
    await actor.close_async()

    assert result == {'documents': [EXAMPLE_DOCUMENT]}


def test_actor_call_timed_out_run_raises() -> None:
    """A run that hits `timeout_secs` raises a `RuntimeError` with the `TIMED-OUT` status."""
    actor = ApifyDatasetFromActorCall(CHEERIO_SCRAPER, run_input=CHEERIO_INPUT, memory_mbytes=1024, timeout_secs=1)

    with pytest.raises(RuntimeError, match='finished with status TIMED-OUT'):
        actor.run()


def test_task_call_loads_run_dataset(task_id: str) -> None:
    """The component runs a saved task with its stored input and loads its dataset."""
    task = ApifyDatasetFromTaskCall(task_id, task_input={'maxPagesPerCrawl': 1})

    result = task.run()
    task.close()

    assert result == {'documents': [EXAMPLE_DOCUMENT]}


async def test_task_call_run_async_loads_run_dataset(task_id: str) -> None:
    """`run_async` runs the task through the async client and loads the same documents."""
    task = ApifyDatasetFromTaskCall(task_id, task_input={'maxPagesPerCrawl': 1})

    result = await task.run_async()
    await task.close_async()

    assert result == {'documents': [EXAMPLE_DOCUMENT]}
