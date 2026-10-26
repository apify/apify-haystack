from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock

import pytest
from apify_client._models import Run
from haystack import Document, Pipeline
from haystack.utils import Secret

from apify_haystack import (
    ApifyDatasetFromActorCall,
    ApifyDatasetFromTaskCall,
    ApifyDatasetLoader,
    _client,
    default_dataset_mapping_function,
)

if TYPE_CHECKING:
    from collections.abc import Callable

TOKEN = Secret.from_token('token')

ITEMS = [
    {'text': 'first', 'url': 'https://example.com/1'},
    {'text': 'second', 'url': 'https://example.com/2'},
]


def to_document(item: dict) -> Document:
    return Document(content=item['text'], meta={'url': item['url']})


@pytest.fixture
def client_class(monkeypatch: pytest.MonkeyPatch) -> Mock:
    client = Mock()
    client.http_client._headers = {'User-Agent': 'ApifyClient/test'}
    client.dataset.return_value.list_items.return_value.items = ITEMS
    client.actor.return_value.call.return_value = Mock(
        spec=Run, id='run-id', status='SUCCEEDED', default_dataset_id='dataset-id'
    )
    client.task.return_value.call.return_value = Mock(
        spec=Run, id='run-id', status='SUCCEEDED', default_dataset_id='dataset-id'
    )
    client_class = Mock(return_value=client)
    monkeypatch.setattr(_client, 'ApifyClient', client_class)
    return client_class


@pytest.fixture
def client(client_class: Mock) -> Mock:
    return client_class.return_value


@pytest.fixture
def async_client_class(monkeypatch: pytest.MonkeyPatch) -> Mock:
    client = Mock()
    client.http_client._headers = {'User-Agent': 'ApifyClient/test'}
    client.http_client.aclose = AsyncMock()
    client.dataset.return_value.list_items = AsyncMock(return_value=Mock(items=ITEMS))
    client.actor.return_value.call = AsyncMock(
        return_value=Mock(spec=Run, id='run-id', status='SUCCEEDED', default_dataset_id='dataset-id')
    )
    client.task.return_value.call = AsyncMock(
        return_value=Mock(spec=Run, id='run-id', status='SUCCEEDED', default_dataset_id='dataset-id')
    )
    async_client_class = Mock(return_value=client)
    monkeypatch.setattr(_client, 'ApifyClientAsync', async_client_class)
    return async_client_class


@pytest.fixture
def async_client(async_client_class: Mock) -> Mock:
    return async_client_class.return_value


def test_loader_maps_dataset_items_to_documents(client: Mock) -> None:
    """The loader lists the clean dataset items and maps each one to a `Document`."""
    loader = ApifyDatasetLoader(dataset_id='dataset-id', dataset_mapping_function=to_document)

    result = loader.run()

    client.dataset.assert_called_once_with('dataset-id')
    client.dataset.return_value.list_items.assert_called_once_with(clean=True)
    assert result == {'documents': [to_document(item) for item in ITEMS]}


@pytest.mark.parametrize(
    'component_class',
    [
        pytest.param(ApifyDatasetFromActorCall, id='actor call'),
        pytest.param(ApifyDatasetFromTaskCall, id='task call'),
    ],
)
def test_missing_token_raises(monkeypatch: pytest.MonkeyPatch, component_class: type) -> None:
    """Without a token argument, `APIFY_TOKEN`, or `APIFY_API_TOKEN`, `warm_up` raises a `ValueError`."""
    monkeypatch.delenv('APIFY_TOKEN', raising=False)
    monkeypatch.delenv('APIFY_API_TOKEN', raising=False)
    instance = component_class('id', to_document)

    with pytest.raises(ValueError, match='APIFY_TOKEN'):
        instance.warm_up()


@pytest.mark.parametrize(
    ('environment', 'expected_token'),
    [
        pytest.param({'APIFY_TOKEN': 'new-token'}, 'new-token', id='APIFY_TOKEN'),
        pytest.param({'APIFY_API_TOKEN': 'old-token'}, 'old-token', id='APIFY_API_TOKEN'),
        pytest.param({'APIFY_TOKEN': 'new-token', 'APIFY_API_TOKEN': 'old-token'}, 'new-token', id='both'),
    ],
)
def test_token_is_read_from_environment(
    monkeypatch: pytest.MonkeyPatch, client_class: Mock, environment: dict[str, str], expected_token: str
) -> None:
    """Without a token argument, the client reads `APIFY_TOKEN`, falling back to the older `APIFY_API_TOKEN`."""
    monkeypatch.delenv('APIFY_TOKEN', raising=False)
    monkeypatch.delenv('APIFY_API_TOKEN', raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    ApifyDatasetFromActorCall('actor-id', to_document).warm_up()

    client_class.assert_called_once_with(expected_token, headers={'x-apify-integration-platform': 'haystack'})


def test_actor_call_runs_actor_and_loads_its_dataset(client: Mock) -> None:
    """The Actor is called with the merged input and run options, and its default dataset is loaded."""
    actor = ApifyDatasetFromActorCall(
        'actor-id',
        to_document,
        run_input={'a': 1},
        build='beta',
        memory_mbytes=512,
        timeout_secs=60,
        apify_token=TOKEN,
    )

    result = actor.run(run_input={'b': 2})

    client.actor.assert_called_once_with('actor-id')
    client.actor.return_value.call.assert_called_once_with(
        run_input={'a': 1, 'b': 2},
        build='beta',
        memory_mbytes=512,
        run_timeout=timedelta(seconds=60),
    )
    client.dataset.assert_called_once_with('dataset-id')
    assert result == {'documents': [to_document(item) for item in ITEMS]}


def test_actor_call_run_input_overrides_constructor_input(client: Mock) -> None:
    """Keys passed to `run` override the same keys from the constructor input."""
    actor = ApifyDatasetFromActorCall('actor-id', to_document, run_input={'a': 1, 'b': 1}, apify_token=TOKEN)

    actor.run(run_input={'b': 2})

    assert client.actor.return_value.call.call_args.kwargs['run_input'] == {'a': 1, 'b': 2}


def test_actor_call_without_input_raises(client: Mock) -> None:
    """Running an Actor without input in either the constructor or `run` raises a `ValueError`."""
    actor = ApifyDatasetFromActorCall('actor-id', to_document, apify_token=TOKEN)

    with pytest.raises(ValueError, match='Please provide the run_input'):
        actor.run()

    client.actor.assert_not_called()


def test_actor_call_without_run_raises(client: Mock) -> None:
    """A run missing from the response raises a `ValueError`."""
    client.actor.return_value.call.return_value = None
    actor = ApifyDatasetFromActorCall('actor-id', to_document, run_input={'a': 1}, apify_token=TOKEN)

    with pytest.raises(ValueError, match='No response found'):
        actor.run()


@pytest.mark.parametrize(
    'status',
    [
        pytest.param('FAILED', id='failed'),
        pytest.param('TIMED-OUT', id='timed out'),
        pytest.param('ABORTED', id='aborted'),
    ],
)
@pytest.mark.parametrize(
    ('component', 'resource'),
    [
        pytest.param(
            lambda: ApifyDatasetFromActorCall('actor-id', to_document, run_input={'a': 1}, apify_token=TOKEN),
            'actor',
            id='actor call',
        ),
        pytest.param(
            lambda: ApifyDatasetFromTaskCall('task-id', to_document, task_input={'a': 1}, apify_token=TOKEN),
            'task',
            id='task call',
        ),
    ],
)
def test_unsuccessful_run_raises(client: Mock, component: Callable[[], Any], resource: str, status: str) -> None:
    """A run that does not succeed raises a `RuntimeError` with its status and loads no dataset."""
    getattr(client, resource).return_value.call.return_value = Mock(spec=Run, id='run-id', status=status)

    with pytest.raises(RuntimeError, match=f'run run-id finished with status {status}'):
        component().run()

    client.dataset.assert_not_called()


async def test_unsuccessful_run_async_raises(async_client: Mock) -> None:
    """In `run_async`, a run that does not succeed raises a `RuntimeError` with its status."""
    async_client.actor.return_value.call.return_value = Mock(spec=Run, id='run-id', status='FAILED')
    actor = ApifyDatasetFromActorCall('actor-id', to_document, run_input={'a': 1}, apify_token=TOKEN)

    with pytest.raises(RuntimeError, match='Actor actor-id run run-id finished with status FAILED'):
        await actor.run_async()

    async_client.dataset.assert_not_called()


def test_actor_call_without_timeout_passes_no_run_timeout(client: Mock) -> None:
    """Without `timeout_secs`, the run uses the Actor's default timeout."""
    actor = ApifyDatasetFromActorCall('actor-id', to_document, run_input={'a': 1}, apify_token=TOKEN)

    actor.run()

    assert client.actor.return_value.call.call_args.kwargs['run_timeout'] is None


def test_task_call_runs_task_and_loads_its_dataset(client: Mock) -> None:
    """The task is called with its input and run options, and its default dataset is loaded."""
    task = ApifyDatasetFromTaskCall(
        'task-id',
        to_document,
        task_input={'a': 1},
        build='beta',
        memory_mbytes=512,
        timeout_secs=60,
        apify_token=TOKEN,
    )

    result = task.run()

    client.task.assert_called_once_with('task-id')
    client.task.return_value.call.assert_called_once_with(
        task_input={'a': 1},
        build='beta',
        memory_mbytes=512,
        run_timeout=timedelta(seconds=60),
    )
    client.dataset.assert_called_once_with('dataset-id')
    assert result == {'documents': [to_document(item) for item in ITEMS]}


@pytest.mark.parametrize(
    ('constructor_input', 'run_input', 'expected_input'),
    [
        pytest.param(None, {'a': 1}, {'a': 1}, id='run input only'),
        pytest.param({'a': 1, 'b': 1}, {'b': 2}, {'a': 1, 'b': 2}, id='run input overrides constructor input'),
    ],
)
def test_task_call_uses_run_input(
    client: Mock, constructor_input: dict | None, run_input: dict, expected_input: dict
) -> None:
    """Input passed to `run` is merged into the constructor input and sent to the task."""
    task = ApifyDatasetFromTaskCall('task-id', to_document, task_input=constructor_input, apify_token=TOKEN)

    task.run(task_input=run_input)

    assert client.task.return_value.call.call_args.kwargs['task_input'] == expected_input


def test_task_call_without_run_raises(client: Mock) -> None:
    """A task run missing from the response raises a `ValueError`."""
    client.task.return_value.call.return_value = None
    task = ApifyDatasetFromTaskCall('task-id', to_document, task_input={'a': 1}, apify_token=TOKEN)

    with pytest.raises(ValueError, match='No response found'):
        task.run()


def test_task_call_without_input_raises(client: Mock) -> None:
    """Running a task without input in either the constructor or `run` raises a `ValueError`."""
    task = ApifyDatasetFromTaskCall('task-id', to_document, apify_token=TOKEN)

    with pytest.raises(ValueError, match='Please provide the task_input'):
        task.run()

    client.task.assert_not_called()


@pytest.mark.parametrize(
    'component',
    [
        pytest.param(lambda: ApifyDatasetLoader('dataset-id', to_document), id='dataset loader'),
        pytest.param(lambda: ApifyDatasetFromActorCall('id', to_document, apify_token=TOKEN), id='actor call'),
        pytest.param(lambda: ApifyDatasetFromTaskCall('id', to_document, apify_token=TOKEN), id='task call'),
    ],
)
def test_client_identifies_haystack_integration(component: Callable[[], Any]) -> None:
    """Every component's client sends the Haystack integration header and origin in its user agent."""
    instance = component()
    instance.warm_up()

    headers = instance.client.http_client._headers

    assert headers['x-apify-integration-platform'] == 'haystack'
    assert headers['User-Agent'].startswith('ApifyClient/')
    assert headers['User-Agent'].endswith('; Origin/haystack')


async def test_async_client_identifies_haystack_integration() -> None:
    """The async client sends the same Haystack integration header and user-agent origin as the sync one."""
    loader = ApifyDatasetLoader('dataset-id', to_document)
    await loader.warm_up_async()

    assert loader.async_client is not None
    headers = loader.async_client.http_client._headers

    assert headers['x-apify-integration-platform'] == 'haystack'
    assert headers['User-Agent'].endswith('; Origin/haystack')


def test_actor_call_runs_in_pipeline(client: Mock) -> None:
    """The component runs inside a Haystack `Pipeline` and takes `run_input` as a pipeline input."""
    pipeline = Pipeline()
    pipeline.add_component('loader', ApifyDatasetFromActorCall('actor-id', to_document, apify_token=TOKEN))

    result = pipeline.run({'loader': {'run_input': {'a': 1}}})

    assert client.actor.return_value.call.call_args.kwargs['run_input'] == {'a': 1}
    assert result == {'loader': {'documents': [to_document(item) for item in ITEMS]}}


def test_client_is_created_on_first_run(client_class: Mock) -> None:
    """The constructor creates no client, and `run` creates one on its first call and reuses it."""
    loader = ApifyDatasetLoader('dataset-id', to_document)
    client_class.assert_not_called()

    loader.run()
    loader.run()

    client_class.assert_called_once()
    assert loader.client is client_class.return_value


def test_close_releases_client(client_class: Mock, client: Mock) -> None:
    """`close` closes the client's HTTP client, and the next `warm_up` creates a new client."""
    loader = ApifyDatasetLoader('dataset-id', to_document)
    loader.warm_up()

    loader.close()

    client.http_client.close.assert_called_once()
    assert loader.client is None
    loader.warm_up()
    assert client_class.call_count == 2


def test_loader_without_token_uses_anonymous_client(monkeypatch: pytest.MonkeyPatch, client_class: Mock) -> None:
    """Without a token argument or a token environment variable, the loader creates a client without a token."""
    monkeypatch.delenv('APIFY_TOKEN', raising=False)
    monkeypatch.delenv('APIFY_API_TOKEN', raising=False)

    ApifyDatasetLoader('dataset-id', to_document).warm_up()

    client_class.assert_called_once_with(None, headers={'x-apify-integration-platform': 'haystack'})


def test_actor_call_to_dict() -> None:
    """`to_dict` stores the mapping function by its import path and the token by its environment variable."""
    actor = ApifyDatasetFromActorCall('actor-id', to_document, run_input={'a': 1}, build='beta', timeout_secs=60)

    data = actor.to_dict()

    assert data == {
        'type': 'apify_haystack.apify_dataset.ApifyDatasetFromActorCall',
        'init_parameters': {
            'actor_id': 'actor-id',
            'dataset_mapping_function': f'{__name__}.to_document',
            'run_input': {'a': 1},
            'build': 'beta',
            'memory_mbytes': None,
            'timeout_secs': 60,
            'apify_token': {'type': 'env_var', 'env_vars': ['APIFY_TOKEN', 'APIFY_API_TOKEN'], 'strict': True},
        },
    }


@pytest.mark.parametrize(
    'component',
    [
        pytest.param(ApifyDatasetLoader('dataset-id', to_document), id='dataset loader'),
        pytest.param(
            ApifyDatasetFromActorCall('actor-id', to_document, run_input={'a': 1}, memory_mbytes=512), id='actor call'
        ),
        pytest.param(
            ApifyDatasetFromTaskCall('task-id', to_document, task_input={'a': 1}, timeout_secs=60), id='task call'
        ),
    ],
)
def test_from_dict_restores_component(monkeypatch: pytest.MonkeyPatch, component: Any) -> None:
    """A component restored from its `to_dict` output has the same configuration."""
    monkeypatch.setenv('HAYSTACK_DESERIALIZATION_ALLOWLIST', __name__)

    restored = type(component).from_dict(component.to_dict())

    assert restored.to_dict() == component.to_dict()
    assert restored.dataset_mapping_function is to_document
    assert isinstance(restored.apify_token, Secret)


def test_pipeline_round_trips_through_yaml() -> None:
    """A pipeline with the component survives `dumps` and `loads` once `apify_haystack` is allowed."""
    pipeline = Pipeline()
    pipeline.add_component('loader', ApifyDatasetFromActorCall('actor-id', to_document, run_input={'a': 1}))

    restored = Pipeline.loads(pipeline.dumps(), allowed_modules=['apify_haystack', __name__])

    assert restored.to_dict() == pipeline.to_dict()


async def test_loader_run_async_maps_dataset_items_to_documents(async_client: Mock, client_class: Mock) -> None:
    """`run_async` lists the dataset with the async client and creates no sync client."""
    loader = ApifyDatasetLoader(dataset_id='dataset-id', dataset_mapping_function=to_document)

    result = await loader.run_async()

    async_client.dataset.return_value.list_items.assert_awaited_once_with(clean=True)
    client_class.assert_not_called()
    assert result == {'documents': [to_document(item) for item in ITEMS]}


async def test_actor_call_run_async_runs_actor_and_loads_its_dataset(async_client: Mock) -> None:
    """`run_async` calls the Actor with the merged input and run options, then loads its default dataset."""
    actor = ApifyDatasetFromActorCall(
        'actor-id', to_document, run_input={'a': 1, 'b': 1}, build='beta', timeout_secs=60, apify_token=TOKEN
    )

    result = await actor.run_async(run_input={'b': 2})

    async_client.actor.return_value.call.assert_awaited_once_with(
        run_input={'a': 1, 'b': 2},
        build='beta',
        memory_mbytes=None,
        run_timeout=timedelta(seconds=60),
    )
    async_client.dataset.assert_called_once_with('dataset-id')
    assert result == {'documents': [to_document(item) for item in ITEMS]}


async def test_task_call_run_async_runs_task_and_loads_its_dataset(async_client: Mock) -> None:
    """`run_async` calls the task with the merged input and run options, then loads its default dataset."""
    task = ApifyDatasetFromTaskCall('task-id', to_document, task_input={'a': 1}, memory_mbytes=512, apify_token=TOKEN)

    result = await task.run_async(task_input={'b': 2})

    async_client.task.return_value.call.assert_awaited_once_with(
        task_input={'a': 1, 'b': 2},
        build=None,
        memory_mbytes=512,
        run_timeout=None,
    )
    assert result == {'documents': [to_document(item) for item in ITEMS]}


@pytest.mark.parametrize(
    ('component', 'match'),
    [
        pytest.param(ApifyDatasetFromActorCall('id', to_document, apify_token=TOKEN), 'run_input', id='actor call'),
        pytest.param(ApifyDatasetFromTaskCall('id', to_document, apify_token=TOKEN), 'task_input', id='task call'),
    ],
)
async def test_run_async_without_input_raises(async_client: Mock, component: Any, match: str) -> None:
    """`run_async` without input in either the constructor or the call raises a `ValueError`."""
    with pytest.raises(ValueError, match=match):
        await component.run_async()

    async_client.actor.assert_not_called()
    async_client.task.assert_not_called()


async def test_run_async_without_run_raises(async_client: Mock) -> None:
    """A run missing from the async response raises a `ValueError`."""
    async_client.actor.return_value.call.return_value = None
    actor = ApifyDatasetFromActorCall('actor-id', to_document, run_input={'a': 1}, apify_token=TOKEN)

    with pytest.raises(ValueError, match='No response found'):
        await actor.run_async()


async def test_close_async_releases_async_client(async_client_class: Mock, async_client: Mock) -> None:
    """`close_async` closes the async HTTP client, and the next `warm_up_async` creates a new client."""
    loader = ApifyDatasetLoader('dataset-id', to_document)
    await loader.warm_up_async()

    await loader.close_async()

    async_client.http_client.aclose.assert_awaited_once()
    assert loader.async_client is None
    await loader.warm_up_async()
    assert async_client_class.call_count == 2


async def test_actor_call_runs_in_async_pipeline(async_client: Mock, client_class: Mock) -> None:
    """`Pipeline.run_async` runs the component through its `run_async` with the async client."""
    pipeline = Pipeline()
    pipeline.add_component('loader', ApifyDatasetFromActorCall('actor-id', to_document, apify_token=TOKEN))

    result = await pipeline.run_async({'loader': {'run_input': {'a': 1}}})

    async_client.actor.return_value.call.assert_awaited_once()
    client_class.assert_not_called()
    assert result == {'loader': {'documents': [to_document(item) for item in ITEMS]}}


@pytest.mark.parametrize(
    ('dataset_item', 'expected'),
    [
        pytest.param(
            {
                'url': 'https://example.com',
                'metadata': {'title': 'Example', 'description': 'An example page'},
                'markdown': '# Example',
                'text': 'Example',
            },
            Document(
                content='# Example',
                meta={'url': 'https://example.com', 'title': 'Example', 'description': 'An example page'},
            ),
            id='website content crawler',
        ),
        pytest.param(
            {'metadata': {'url': 'https://example.com', 'title': 'Example'}, 'text': 'Example'},
            Document(content='Example', meta={'url': 'https://example.com', 'title': 'Example'}),
            id='text without markdown',
        ),
        pytest.param(
            {'searchResult': {'url': 'https://example.com', 'title': 'Result', 'description': 'Snippet'}},
            Document(
                content='{"searchResult": {"url": "https://example.com", "title": "Result", "description": "Snippet"}}',
                meta={'url': 'https://example.com', 'title': 'Result', 'description': 'Snippet'},
            ),
            id='search result only',
        ),
        pytest.param(
            {'name': 'Cafe', 'rating': 4.5, 'metadata': 'not a dict'},
            Document(content='{"name": "Cafe", "rating": 4.5, "metadata": "not a dict"}'),
            id='generic item',
        ),
    ],
)
def test_default_dataset_mapping_function(dataset_item: dict, expected: Document) -> None:
    """The default mapping takes Markdown or text with the page details, and falls back to the item as JSON."""
    assert default_dataset_mapping_function(dataset_item) == expected


@pytest.mark.usefixtures('client')
@pytest.mark.parametrize(
    'component',
    [
        pytest.param(lambda: ApifyDatasetLoader('dataset-id'), id='dataset loader'),
        pytest.param(lambda: ApifyDatasetFromActorCall('id', run_input={'a': 1}, apify_token=TOKEN), id='actor call'),
        pytest.param(lambda: ApifyDatasetFromTaskCall('id', task_input={'a': 1}, apify_token=TOKEN), id='task call'),
    ],
)
def test_components_default_to_default_mapping_function(component: Callable[[], Any]) -> None:
    """Without a mapping function, the components convert items with `default_dataset_mapping_function`."""
    result = component().run()

    assert result == {'documents': [default_dataset_mapping_function(item) for item in ITEMS]}


def test_default_mapping_function_round_trips_through_serialization() -> None:
    """The default mapping function is stored as `None`, so restoring it needs no allowlisted module."""
    loader = ApifyDatasetLoader('dataset-id')

    data = loader.to_dict()
    restored = ApifyDatasetLoader.from_dict(data)

    assert data['init_parameters']['dataset_mapping_function'] is None
    assert restored.dataset_mapping_function is default_dataset_mapping_function
