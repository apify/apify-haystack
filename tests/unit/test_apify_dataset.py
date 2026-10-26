from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest
from apify_client._models import Run
from haystack import Document, Pipeline
from haystack.utils import Secret

from apify_haystack import ApifyDatasetFromActorCall, ApifyDatasetFromTaskCall, ApifyDatasetLoader, apify_dataset

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
    client.actor.return_value.call.return_value = Mock(spec=Run, default_dataset_id='dataset-id')
    client.task.return_value.call.return_value = Mock(spec=Run, default_dataset_id='dataset-id')
    client_class = Mock(return_value=client)
    monkeypatch.setattr(apify_dataset, 'ApifyClient', client_class)
    return client_class


@pytest.fixture
def client(client_class: Mock) -> Mock:
    return client_class.return_value


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
    """Without a token argument or `APIFY_API_TOKEN`, `warm_up` raises a `ValueError`."""
    monkeypatch.delenv('APIFY_API_TOKEN', raising=False)
    instance = component_class('id', to_document)

    with pytest.raises(ValueError, match='APIFY_API_TOKEN'):
        instance.warm_up()


def test_token_is_read_from_environment(monkeypatch: pytest.MonkeyPatch, client_class: Mock) -> None:
    """Without a token argument, the client gets the token from `APIFY_API_TOKEN`."""
    monkeypatch.setenv('APIFY_API_TOKEN', 'env-token')

    ApifyDatasetFromActorCall('actor-id', to_document).warm_up()

    client_class.assert_called_once_with('env-token', headers={'x-apify-integration-platform': 'haystack'})


def test_actor_call_runs_actor_and_loads_its_dataset(client: Mock) -> None:
    """The Actor is called with the merged input and run options, and its default dataset is loaded."""
    actor = ApifyDatasetFromActorCall(
        'actor-id',
        to_document,
        run_input={'a': 1},
        build='beta',
        memory_mbytes=512,
        timeout_secs=60,
        apify_api_token=TOKEN,
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
    actor = ApifyDatasetFromActorCall('actor-id', to_document, run_input={'a': 1, 'b': 1}, apify_api_token=TOKEN)

    actor.run(run_input={'b': 2})

    assert client.actor.return_value.call.call_args.kwargs['run_input'] == {'a': 1, 'b': 2}


def test_actor_call_without_input_raises(client: Mock) -> None:
    """Running an Actor without input in either the constructor or `run` raises a `ValueError`."""
    actor = ApifyDatasetFromActorCall('actor-id', to_document, apify_api_token=TOKEN)

    with pytest.raises(ValueError, match='Please provide the run_input'):
        actor.run()

    client.actor.assert_not_called()


def test_actor_call_without_run_raises(client: Mock) -> None:
    """A run missing from the response raises a `ValueError`."""
    client.actor.return_value.call.return_value = None
    actor = ApifyDatasetFromActorCall('actor-id', to_document, run_input={'a': 1}, apify_api_token=TOKEN)

    with pytest.raises(ValueError, match='No response found'):
        actor.run()


def test_actor_call_without_timeout_passes_no_run_timeout(client: Mock) -> None:
    """Without `timeout_secs`, the run uses the Actor's default timeout."""
    actor = ApifyDatasetFromActorCall('actor-id', to_document, run_input={'a': 1}, apify_api_token=TOKEN)

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
        apify_api_token=TOKEN,
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
    task = ApifyDatasetFromTaskCall('task-id', to_document, task_input=constructor_input, apify_api_token=TOKEN)

    task.run(task_input=run_input)

    assert client.task.return_value.call.call_args.kwargs['task_input'] == expected_input


def test_task_call_without_run_raises(client: Mock) -> None:
    """A task run missing from the response raises a `ValueError`."""
    client.task.return_value.call.return_value = None
    task = ApifyDatasetFromTaskCall('task-id', to_document, task_input={'a': 1}, apify_api_token=TOKEN)

    with pytest.raises(ValueError, match='No response found'):
        task.run()


def test_task_call_without_input_raises(client: Mock) -> None:
    """Running a task without input in either the constructor or `run` raises a `ValueError`."""
    task = ApifyDatasetFromTaskCall('task-id', to_document, apify_api_token=TOKEN)

    with pytest.raises(ValueError, match='Please provide the task_input'):
        task.run()

    client.task.assert_not_called()


@pytest.mark.parametrize(
    'component',
    [
        pytest.param(lambda: ApifyDatasetLoader('dataset-id', to_document), id='dataset loader'),
        pytest.param(lambda: ApifyDatasetFromActorCall('id', to_document, apify_api_token=TOKEN), id='actor call'),
        pytest.param(lambda: ApifyDatasetFromTaskCall('id', to_document, apify_api_token=TOKEN), id='task call'),
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


def test_actor_call_runs_in_pipeline(client: Mock) -> None:
    """The component runs inside a Haystack `Pipeline` and takes `run_input` as a pipeline input."""
    pipeline = Pipeline()
    pipeline.add_component('loader', ApifyDatasetFromActorCall('actor-id', to_document, apify_api_token=TOKEN))

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
    """Without a token argument or `APIFY_API_TOKEN`, the loader creates a client without a token."""
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
            'apify_api_token': {'type': 'env_var', 'env_vars': ['APIFY_API_TOKEN'], 'strict': True},
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
    assert isinstance(restored.apify_api_token, Secret)


def test_pipeline_round_trips_through_yaml() -> None:
    """A pipeline with the component survives `dumps` and `loads` once `apify_haystack` is allowed."""
    pipeline = Pipeline()
    pipeline.add_component('loader', ApifyDatasetFromActorCall('actor-id', to_document, run_input={'a': 1}))

    restored = Pipeline.loads(pipeline.dumps(), allowed_modules=['apify_haystack', __name__])

    assert restored.to_dict() == pipeline.to_dict()
