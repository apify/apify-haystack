from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest
from haystack import Document

from apify_haystack import ApifyDatasetFromActorCall, ApifyDatasetFromTaskCall, ApifyDatasetLoader, apify_dataset

if TYPE_CHECKING:
    from collections.abc import Callable

ITEMS = [
    {'text': 'first', 'url': 'https://example.com/1'},
    {'text': 'second', 'url': 'https://example.com/2'},
]


def to_document(item: dict) -> Document:
    return Document(content=item['text'], meta={'url': item['url']})


@pytest.fixture
def client_class(monkeypatch: pytest.MonkeyPatch) -> Mock:
    client = Mock()
    client.http_client.httpx_client.headers = {'user-agent': 'ApifyClient/test'}
    client.dataset.return_value.list_items.return_value.items = ITEMS
    client.actor.return_value.call.return_value = {'defaultDatasetId': 'dataset-id'}
    client.task.return_value.call.return_value = {'defaultDatasetId': 'dataset-id'}
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
    """Without a token argument or `APIFY_API_TOKEN`, the constructor raises a `ValueError`."""
    monkeypatch.delenv('APIFY_API_TOKEN', raising=False)

    with pytest.raises(ValueError, match='Apify API token not found'):
        component_class('id', to_document)


def test_token_is_read_from_environment(monkeypatch: pytest.MonkeyPatch, client_class: Mock) -> None:
    """Without a token argument, the client gets the token from `APIFY_API_TOKEN`."""
    monkeypatch.setenv('APIFY_API_TOKEN', 'env-token')

    ApifyDatasetFromActorCall('actor-id', to_document)

    client_class.assert_called_once_with('env-token')


def test_actor_call_runs_actor_and_loads_its_dataset(client: Mock) -> None:
    """The Actor is called with the merged input and run options, and its default dataset is loaded."""
    actor = ApifyDatasetFromActorCall(
        'actor-id',
        to_document,
        run_input={'a': 1},
        build='beta',
        memory_mbytes=512,
        timeout_secs=60,
        apify_api_token='token',
    )

    result = actor.run(run_input={'b': 2})

    client.actor.assert_called_once_with('actor-id')
    client.actor.return_value.call.assert_called_once_with(
        run_input={'a': 1, 'b': 2},
        build='beta',
        memory_mbytes=512,
        timeout_secs=60,
    )
    client.dataset.assert_called_once_with('dataset-id')
    assert result == {'documents': [to_document(item) for item in ITEMS]}


def test_actor_call_run_input_overrides_constructor_input(client: Mock) -> None:
    """Keys passed to `run` override the same keys from the constructor input."""
    actor = ApifyDatasetFromActorCall('actor-id', to_document, run_input={'a': 1, 'b': 1}, apify_api_token='token')

    actor.run(run_input={'b': 2})

    assert client.actor.return_value.call.call_args.kwargs['run_input'] == {'a': 1, 'b': 2}


def test_actor_call_without_input_raises(client: Mock) -> None:
    """Running an Actor without input in either the constructor or `run` raises a `ValueError`."""
    actor = ApifyDatasetFromActorCall('actor-id', to_document, apify_api_token='token')

    with pytest.raises(ValueError, match='Please provide the run_input'):
        actor.run()

    client.actor.assert_not_called()


@pytest.mark.parametrize(
    ('call_result', 'match'),
    [
        pytest.param(None, 'No response found', id='no run'),
        pytest.param({'id': 'run-id'}, 'No dataset found', id='no dataset'),
    ],
)
def test_actor_call_with_unusable_run_raises(client: Mock, call_result: dict | None, match: str) -> None:
    """A run missing from the response, or without a default dataset, raises a `ValueError`."""
    client.actor.return_value.call.return_value = call_result
    actor = ApifyDatasetFromActorCall('actor-id', to_document, run_input={'a': 1}, apify_api_token='token')

    with pytest.raises(ValueError, match=match):
        actor.run()


def test_task_call_runs_task_and_loads_its_dataset(client: Mock) -> None:
    """The task is called with its input and run options, and its default dataset is loaded."""
    task = ApifyDatasetFromTaskCall(
        'task-id',
        to_document,
        task_input={'a': 1},
        build='beta',
        memory_mbytes=512,
        timeout_secs=60,
        apify_api_token='token',
    )

    result = task.run()

    client.task.assert_called_once_with('task-id')
    client.task.return_value.call.assert_called_once_with(
        task_input={'a': 1},
        build='beta',
        memory_mbytes=512,
        timeout_secs=60,
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
    task = ApifyDatasetFromTaskCall('task-id', to_document, task_input=constructor_input, apify_api_token='token')

    task.run(task_input=run_input)

    assert client.task.return_value.call.call_args.kwargs['task_input'] == expected_input


def test_task_call_without_input_raises(client: Mock) -> None:
    """Running a task without input in either the constructor or `run` raises a `ValueError`."""
    task = ApifyDatasetFromTaskCall('task-id', to_document, apify_api_token='token')

    with pytest.raises(ValueError, match='Please provide the task_input'):
        task.run()

    client.task.assert_not_called()


@pytest.mark.parametrize(
    'component',
    [
        pytest.param(lambda: ApifyDatasetLoader('dataset-id', to_document), id='dataset loader'),
        pytest.param(lambda: ApifyDatasetFromActorCall('id', to_document, apify_api_token='token'), id='actor call'),
        pytest.param(lambda: ApifyDatasetFromTaskCall('id', to_document, apify_api_token='token'), id='task call'),
    ],
)
def test_user_agent_has_haystack_origin(component: Callable[[], Any]) -> None:
    """Every component's client sends the Haystack origin in its user agent."""
    instance = component()

    user_agent = instance.client.http_client.httpx_client.headers['user-agent']

    assert user_agent.startswith('ApifyClient/')
    assert user_agent.endswith('; Origin/haystack')
