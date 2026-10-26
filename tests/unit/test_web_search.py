from __future__ import annotations

from datetime import timedelta
from unittest.mock import AsyncMock, Mock

import pytest
from apify_client._models import Run
from haystack import Document, Pipeline
from haystack.utils import Secret

from apify_haystack import ApifyWebSearch, apify_dataset

TOKEN = Secret.from_token('token')

ITEMS = [
    {
        'metadata': {'url': 'https://example.com/1', 'title': 'First', 'description': 'First page'},
        'searchResult': {'url': 'https://example.com/1', 'title': 'First result', 'description': 'Snippet'},
        'markdown': '# First',
    },
    {
        'searchResult': {'url': 'https://example.com/2', 'title': 'Second', 'description': 'Not scraped'},
        'text': 'second',
    },
]

DOCUMENTS = [
    Document(content='# First', meta={'url': 'https://example.com/1', 'title': 'First', 'description': 'First page'}),
    Document(content='second', meta={'url': 'https://example.com/2', 'title': 'Second', 'description': 'Not scraped'}),
]


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Mock:
    client = Mock()
    client.http_client._headers = {'User-Agent': 'ApifyClient/test'}
    client.actor.return_value.call.return_value = Mock(
        spec=Run, id='run-id', status='SUCCEEDED', default_dataset_id='dataset-id'
    )
    client.dataset.return_value.list_items.return_value.items = ITEMS
    monkeypatch.setattr(apify_dataset, 'ApifyClient', Mock(return_value=client))
    return client


@pytest.fixture
def async_client(monkeypatch: pytest.MonkeyPatch) -> Mock:
    client = Mock()
    client.http_client._headers = {'User-Agent': 'ApifyClient/test'}
    client.actor.return_value.call = AsyncMock(
        return_value=Mock(spec=Run, id='run-id', status='SUCCEEDED', default_dataset_id='dataset-id')
    )
    client.dataset.return_value.list_items = AsyncMock(return_value=Mock(items=ITEMS))
    monkeypatch.setattr(apify_dataset, 'ApifyClientAsync', Mock(return_value=client))
    return client


def test_run_searches_and_returns_documents_and_links(client: Mock) -> None:
    """`run` calls the RAG Web Browser with the query and returns one `Document` and link per result page."""
    search = ApifyWebSearch(top_k=2, search_params={'scrapingTool': 'raw-http'}, timeout_secs=60, apify_token=TOKEN)

    result = search.run(query='haystack')

    client.actor.assert_called_once_with('apify/rag-web-browser')
    client.actor.return_value.call.assert_called_once_with(
        run_input={'scrapingTool': 'raw-http', 'query': 'haystack', 'maxResults': 2},
        run_timeout=timedelta(seconds=60),
    )
    client.dataset.assert_called_once_with('dataset-id')
    assert result == {'documents': DOCUMENTS, 'links': ['https://example.com/1', 'https://example.com/2']}


def test_search_params_cannot_override_query_or_top_k(client: Mock) -> None:
    """`query` and `top_k` win over the same keys in `search_params`."""
    search = ApifyWebSearch(top_k=3, search_params={'query': 'other', 'maxResults': 50}, apify_token=TOKEN)

    search.run(query='haystack')

    run_input = client.actor.return_value.call.call_args.kwargs['run_input']
    assert run_input == {'query': 'haystack', 'maxResults': 3}


def test_result_without_url_has_no_link(client: Mock) -> None:
    """A result without any URL becomes a `Document` but adds nothing to `links`."""
    client.dataset.return_value.list_items.return_value.items = [{'markdown': 'content'}]

    result = ApifyWebSearch(apify_token=TOKEN).run(query='haystack')

    assert result == {
        'documents': [Document(content='content')],
        'links': [],
    }


def test_run_without_run_raises(client: Mock) -> None:
    """A run missing from the response raises a `ValueError`."""
    client.actor.return_value.call.return_value = None

    with pytest.raises(ValueError, match='No response found'):
        ApifyWebSearch(apify_token=TOKEN).run(query='haystack')


def test_unsuccessful_run_raises(client: Mock) -> None:
    """A search run that does not succeed raises a `RuntimeError` with its status."""
    client.actor.return_value.call.return_value = Mock(spec=Run, id='run-id', status='TIMED-OUT')

    with pytest.raises(RuntimeError, match='apify/rag-web-browser run run-id finished with status TIMED-OUT'):
        ApifyWebSearch(apify_token=TOKEN).run(query='haystack')

    client.dataset.assert_not_called()


@pytest.mark.parametrize(
    'top_k',
    [
        pytest.param(0, id='zero'),
        pytest.param(101, id='above maximum'),
    ],
)
def test_top_k_out_of_range_raises(top_k: int) -> None:
    """A `top_k` outside the range the Actor accepts raises a `ValueError`."""
    with pytest.raises(ValueError, match='top_k must be between 1 and 100'):
        ApifyWebSearch(top_k=top_k, apify_token=TOKEN)


async def test_run_async_searches_and_returns_documents_and_links(async_client: Mock) -> None:
    """`run_async` calls the RAG Web Browser with the async client and returns the same output as `run`."""
    search = ApifyWebSearch(top_k=2, apify_token=TOKEN)

    result = await search.run_async(query='haystack')

    async_client.actor.return_value.call.assert_awaited_once_with(
        run_input={'query': 'haystack', 'maxResults': 2},
        run_timeout=None,
    )
    assert result == {'documents': DOCUMENTS, 'links': ['https://example.com/1', 'https://example.com/2']}


def test_runs_in_pipeline(client: Mock) -> None:
    """The component runs inside a Haystack `Pipeline` and takes `query` as a pipeline input."""
    pipeline = Pipeline()
    pipeline.add_component('search', ApifyWebSearch(apify_token=TOKEN))

    result = pipeline.run({'search': {'query': 'haystack'}})

    assert client.actor.return_value.call.call_args.kwargs['run_input']['query'] == 'haystack'
    assert result['search']['documents'] == DOCUMENTS


def test_to_dict() -> None:
    """`to_dict` stores the search options and the token by its environment variable."""
    search = ApifyWebSearch(top_k=3, search_params={'scrapingTool': 'raw-http'}, timeout_secs=60)

    assert search.to_dict() == {
        'type': 'apify_haystack.web_search.ApifyWebSearch',
        'init_parameters': {
            'top_k': 3,
            'search_params': {'scrapingTool': 'raw-http'},
            'timeout_secs': 60,
            'apify_token': {'type': 'env_var', 'env_vars': ['APIFY_TOKEN', 'APIFY_API_TOKEN'], 'strict': True},
        },
    }


def test_from_dict_restores_component() -> None:
    """A component restored from its `to_dict` output has the same configuration."""
    search = ApifyWebSearch(top_k=3, search_params={'scrapingTool': 'raw-http'})

    restored = ApifyWebSearch.from_dict(search.to_dict())

    assert restored.to_dict() == search.to_dict()
    assert isinstance(restored.apify_token, Secret)
