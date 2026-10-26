from __future__ import annotations

import os
from datetime import timedelta
from typing import TYPE_CHECKING

from apify_client import ApifyClient
from haystack import component
from haystack.dataclasses import Document

from apify_haystack.constants import HAYSTACK_ATTRIBUTE_USER_AGENT, HAYSTACK_INTEGRATION_PLATFORM

if TYPE_CHECKING:
    from collections.abc import Callable


class _ApifyClientComponent:
    """Base for the components that call the Apify API. The client lives between `warm_up` and `close`."""

    client: ApifyClient | None

    def warm_up(self) -> None:
        """Create the Apify client. `run` does this itself when the component was not warmed up."""
        self._get_client()

    def close(self) -> None:
        """Release the Apify client. The next `warm_up` or `run` creates a new one."""
        if self.client is not None:
            self.client.http_client.close()
            self.client = None

    def _get_client(self) -> ApifyClient:
        if self.client is None:
            self.client = _create_client(self._resolve_token())
        return self.client

    def _resolve_token(self) -> str | None:
        return None


@component
class ApifyDatasetLoader(_ApifyClientComponent):
    """Load datasets produced by `Apify Actors`.

    Actors are serverless functions that can be used to scrape, crawl, and extract data from the web.
    For details, see https://docs.apify.com/platform/integrations/haystack

    Example:
        .. code-block:: python

            from haystack import Document

            from apify_haystack import ApifyDatasetLoader

            def dataset_mapping_function(dataset_item):
                return Document(content=dataset_item.get("text"), meta={"url": dataset_item.get("url")})

            loader = ApifyDatasetLoader(dataset_id=dataset_id, dataset_mapping_function=dataset_mapping_function)
            dataset = loader.run()
    """

    def __init__(
        self,
        dataset_id: str,
        dataset_mapping_function: Callable[[dict], Document],
    ) -> None:
        """Initialize the Apify dataset loader.

        Args:
            dataset_id (str): The ID of the dataset on the Apify platform.
            dataset_mapping_function (Callable): A function that takes a single (an Apify dataset item) and converts
                it to an instance of the Document class.
        """
        self.dataset_id = dataset_id
        self.dataset_mapping_function = dataset_mapping_function
        self.client = None

    @component.output_types(documents=list[Document])
    def run(self) -> dict[str, list[Document]]:
        """Load datasets produced by the `Apify Actors`."""
        return {'documents': _load_documents(self._get_client(), self.dataset_id, self.dataset_mapping_function)}


@component
class ApifyDatasetFromActorCall(_ApifyClientComponent):
    """Get an Apify dataset by calling an Apify Actor.

    Perform call of Apify Actor and load produced dataset.
    For details, see https://docs.apify.com/platform/integrations/haystack

    To use it, you should have the ``apify-client`` python package installed,
    and the environment variable ``APIFY_API_TOKEN`` set with your API key, or pass
    `apify_api_token` as a named parameter to the constructor.
    """

    def __init__(
        self,
        actor_id: str,
        dataset_mapping_function: Callable[[dict], Document],
        run_input: dict | None = None,
        *,
        build: str | None = None,
        memory_mbytes: int | None = None,
        timeout_secs: int | None = None,
        apify_api_token: str | None = None,
    ) -> None:
        """Initialize the Apify Actor loader.

        Args:
            actor_id (str): The ID or name of the Actor on the Apify platform.
            dataset_mapping_function (Callable): A function that takes a single
                dictionary (an Apify dataset item) and converts it to an
                instance of the Document class.
            run_input (dict, optional): The input parameters for the Actor you want to run. This can be provided
                either in the `run` method or during the class instantiation. If specified in both places,
                the inputs are merged and the values from `run` take precedence.
            build (str, optional): Optionally specifies the actor build to run.
                It can be either a build tag or build number.
            memory_mbytes (int, optional): Optional memory limit for the run, in megabytes.
            timeout_secs (int, optional): Optional timeout for the run, in seconds.
            apify_api_token (str, optional): The API token for the Apify platform.
                If not provided, it will be read from the environment
        """
        self.actor_id = actor_id
        self.dataset_mapping_function = dataset_mapping_function
        self.run_input = run_input
        self.build = build
        self.memory_mbytes = memory_mbytes
        self.timeout_secs = timeout_secs
        self.apify_api_token = apify_api_token
        self.client = None

    @component.output_types(documents=list[Document])
    def run(self, run_input: dict | None = None) -> dict[str, list[Document]]:
        """Run an Actor on the Apify platform and wait for results to be ready.

        Args:
            run_input (dict, optional): The input parameters for the Actor you want to run. This can be provided
                either in the `run` method or during the class instantiation. If specified in both places,
                the inputs are merged and the values from `run` take precedence.

        Example:
            .. code-block:: python

                actor = ApifyDatasetFromActorCall(
                    actor_id="apify/website-content-crawler",
                    run_input={ "maxCrawlPages": 5 }
                )
                dataset = actor.run(run_input={ "startUrls": [{"url": "https://haystack.deepset.ai/"}] })
        """
        if not (run_input := _merge_inputs(self.run_input, run_input)):
            raise ValueError('Please provide the run_input either in the constructor or in the run method.')

        client = self._get_client()
        run = client.actor(self.actor_id).call(
            run_input=run_input,
            build=self.build,
            memory_mbytes=self.memory_mbytes,
            run_timeout=_to_timedelta(self.timeout_secs),
        )
        if run is None:
            raise ValueError(f'No response found in the actor {self.actor_id} call')

        return {'documents': _load_documents(client, run.default_dataset_id, self.dataset_mapping_function)}

    def _resolve_token(self) -> str:
        return _require_token(self.apify_api_token, type(self).__name__)


@component
class ApifyDatasetFromTaskCall(_ApifyClientComponent):
    """Get an Apify dataset by calling an Apify task.

    Perform a call of an Apify task and load the produced dataset.
    """

    def __init__(
        self,
        task_id: str,
        dataset_mapping_function: Callable[[dict], Document],
        task_input: dict | None = None,
        *,
        build: str | None = None,
        memory_mbytes: int | None = None,
        timeout_secs: int | None = None,
        apify_api_token: str | None = None,
    ) -> None:
        """Initialize the Apify Actor loader.

        Args:
            task_id (str): The ID or name of the Actor on the Apify platform.
            dataset_mapping_function (Callable): A function that takes a single
                dictionary (an Apify dataset item) and converts it to an
                instance of the Document class.
            task_input (dict, optional): The input parameters for the Actor you want to run. This can be provided
                either in the `run` method or during the class instantiation. If specified in both places,
                the inputs are merged and the values from `run` take precedence.
            build (str, optional): Optionally specifies the Actor build to run.
                It can be either a build tag or build number.
            memory_mbytes (int, optional): Optional memory limit for the run, in megabytes.
            timeout_secs (int, optional): Optional timeout for the run, in seconds.
            apify_api_token (str, optional): The API token for the Apify platform.
                If not provided, it will be read from the environment
        """
        self.task_id = task_id
        self.dataset_mapping_function = dataset_mapping_function
        self.task_input = task_input
        self.build = build
        self.memory_mbytes = memory_mbytes
        self.timeout_secs = timeout_secs
        self.apify_api_token = apify_api_token
        self.client = None

    @component.output_types(documents=list[Document])
    def run(self, task_input: dict | None = None) -> dict[str, list[Document]]:
        """Run an Actor on the Apify platform and wait for results to be ready.

        Args:
            task_input (dict, optional): The input parameters for the Actor you want to run. This can be provided
                either in the `run` method or during the class instantiation. If specified in both places,
                the inputs are merged and the values from `run` take precedence.
        """
        if not (task_input := _merge_inputs(self.task_input, task_input)):
            raise ValueError('Please provide the task_input either in the constructor or in the run method.')

        client = self._get_client()
        run = client.task(self.task_id).call(
            task_input=task_input,
            build=self.build,
            memory_mbytes=self.memory_mbytes,
            run_timeout=_to_timedelta(self.timeout_secs),
        )
        if run is None:
            raise ValueError(f'No response found in the task {self.task_id} call')

        return {'documents': _load_documents(client, run.default_dataset_id, self.dataset_mapping_function)}

    def _resolve_token(self) -> str:
        return _require_token(self.apify_api_token, type(self).__name__)


def _merge_inputs(input1: dict | None, input2: dict | None) -> dict | None:
    if input1 and input2:
        return input1 | input2
    return input1 or input2


def _require_token(token: str | None, component_name: str) -> str:
    if not (token := token or os.getenv('APIFY_API_TOKEN')):
        raise ValueError(
            f'Apify API token not found. Please provide Apify API token when initializing {component_name}, '
            'or set the environment variable APIFY_API_TOKEN.'
        )
    return token


def _create_client(token: str | None = None) -> ApifyClient:
    """Create an Apify client that identifies its requests as coming from this integration."""
    client = ApifyClient(token, headers={'x-apify-integration-platform': HAYSTACK_INTEGRATION_PLATFORM})

    # The user-agent marker predates the integration platform header and is kept for older platform paths. Passing it
    # through `headers` would replace apify-client's own user agent, so the header is extended in place.
    client.http_client._headers['User-Agent'] += f'; {HAYSTACK_ATTRIBUTE_USER_AGENT}'  # noqa: SLF001
    return client


def _load_documents(
    client: ApifyClient,
    dataset_id: str,
    dataset_mapping_function: Callable[[dict], Document],
) -> list[Document]:
    dataset_items = client.dataset(dataset_id).list_items(clean=True).items
    return [dataset_mapping_function(item) for item in dataset_items]


def _to_timedelta(seconds: int | None) -> timedelta | None:
    return None if seconds is None else timedelta(seconds=seconds)
