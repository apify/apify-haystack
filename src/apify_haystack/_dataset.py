from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Self

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.utils import Secret, deserialize_callable, serialize_callable

from apify_haystack._client import ApifyClientComponent, get_dataset_id, to_decimal, to_timedelta

if TYPE_CHECKING:
    from collections.abc import Callable

    from apify_client import ApifyClient, ApifyClientAsync


class ApifyDatasetComponent(ApifyClientComponent):
    """Base for the components that convert dataset items to Documents with a mapping function."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize the component from a dictionary created by `to_dict`."""
        init_parameters = data['init_parameters']
        serialized_function = init_parameters.get('dataset_mapping_function')
        mapping_function = deserialize_callable(serialized_function) if serialized_function else None
        data = {**data, 'init_parameters': {**init_parameters, 'dataset_mapping_function': mapping_function}}
        return default_from_dict(cls, data)


@component
class ApifyDatasetLoader(ApifyDatasetComponent):
    """Load datasets produced by Apify Actors.

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

    # Serialized pipelines store the class by this public path, so it does not depend on the module layout.
    __module__ = 'apify_haystack'

    def __init__(
        self,
        dataset_id: str,
        dataset_mapping_function: Callable[[dict], Document] | None = None,
        *,
        apify_token: Secret = Secret.from_env_var(['APIFY_TOKEN', 'APIFY_API_TOKEN'], strict=False),
    ) -> None:
        """Initialize the Apify dataset loader.

        Args:
            dataset_id (str): The ID of the dataset on the Apify platform.
            dataset_mapping_function (Callable, optional): A function that takes a single dictionary (an Apify dataset
                item) and converts it to an instance of the Document class. Defaults to
                `default_dataset_mapping_function`.
            apify_token (Secret, optional): The API token for the Apify platform. Defaults to the `APIFY_TOKEN`
                environment variable, or the older `APIFY_API_TOKEN`. Without a token, only public datasets can be
                loaded.
        """
        self.dataset_id = dataset_id
        self.dataset_mapping_function = dataset_mapping_function or default_dataset_mapping_function
        self.apify_token = apify_token
        self.client = None
        self.async_client = None

    @component.output_types(documents=list[Document])
    def run(self) -> dict[str, list[Document]]:
        """Load the dataset produced by an Apify Actor."""
        return {'documents': _load_documents(self._get_client(), self.dataset_id, self.dataset_mapping_function)}

    @component.output_types(documents=list[Document])
    async def run_async(self) -> dict[str, list[Document]]:
        """Load the dataset produced by an Apify Actor, asynchronously."""
        client = self._get_async_client()
        return {'documents': await _load_documents_async(client, self.dataset_id, self.dataset_mapping_function)}

    def to_dict(self) -> dict[str, Any]:
        """Serialize the component to a dictionary."""
        return default_to_dict(
            self,
            dataset_id=self.dataset_id,
            dataset_mapping_function=_serialize_mapping_function(self.dataset_mapping_function),
            apify_token=self.apify_token,
        )


@component
class ApifyDatasetFromActorCall(ApifyDatasetComponent):
    """Get an Apify dataset by calling an Apify Actor.

    Perform a call of an Apify Actor and load the produced dataset.
    For details, see https://docs.apify.com/platform/integrations/haystack

    It needs an Apify API token, read from the `APIFY_TOKEN` environment variable by default, or passed as
    `apify_token`.
    """

    # Serialized pipelines store the class by this public path, so it does not depend on the module layout.
    __module__ = 'apify_haystack'

    def __init__(
        self,
        actor_id: str,
        dataset_mapping_function: Callable[[dict], Document] | None = None,
        run_input: dict | None = None,
        *,
        build: str | None = None,
        memory_mbytes: int | None = None,
        timeout_secs: int | None = None,
        max_total_charge_usd: float | None = None,
        apify_token: Secret = Secret.from_env_var(['APIFY_TOKEN', 'APIFY_API_TOKEN']),
    ) -> None:
        """Initialize the Apify Actor loader.

        Args:
            actor_id (str): The ID or name of the Actor on the Apify platform.
            dataset_mapping_function (Callable, optional): A function that takes a single dictionary (an Apify dataset
                item) and converts it to an instance of the Document class. Defaults to
                `default_dataset_mapping_function`.
            run_input (dict, optional): The input parameters for the Actor you want to run. This can be provided
                either in the `run` method or during the class instantiation. If specified in both places,
                the inputs are merged and the values from `run` take precedence. Without any input, the
                Actor runs with its default input.
            build (str, optional): Optionally specifies the Actor build to run.
                It can be either a build tag or build number.
            memory_mbytes (int, optional): Optional memory limit for the run, in megabytes.
            timeout_secs (int, optional): Optional timeout for the run, in seconds.
            max_total_charge_usd (float, optional): The maximum amount in USD to charge for the run of a pay-per-event
                Actor. The platform charges no more than that.
            apify_token (Secret, optional): The API token for the Apify platform. Defaults to the
                `APIFY_TOKEN` environment variable, or the older `APIFY_API_TOKEN`.
        """
        self.actor_id = actor_id
        self.dataset_mapping_function = dataset_mapping_function or default_dataset_mapping_function
        self.run_input = run_input
        self.build = build
        self.memory_mbytes = memory_mbytes
        self.timeout_secs = timeout_secs
        self.max_total_charge_usd = max_total_charge_usd
        self.apify_token = apify_token
        self.client = None
        self.async_client = None

    @component.output_types(documents=list[Document])
    def run(self, run_input: dict | None = None) -> dict[str, list[Document]]:
        """Run an Actor on the Apify platform and wait for results to be ready.

        Args:
            run_input (dict, optional): The input parameters for the Actor you want to run. This can be provided
                either in the `run` method or during the class instantiation. If specified in both places,
                the inputs are merged and the values from `run` take precedence. Without any input, the
                Actor runs with its default input.

        Example:
            .. code-block:: python

                actor = ApifyDatasetFromActorCall(
                    actor_id="apify/website-content-crawler",
                    dataset_mapping_function=lambda item: Document(content=item.get("text")),
                    run_input={"maxCrawlPages": 5},
                )
                dataset = actor.run(run_input={"startUrls": [{"url": "https://haystack.deepset.ai/"}]})
        """
        run_input = _merge_inputs(self.run_input, run_input)

        client = self._get_client()
        run = client.actor(self.actor_id).call(
            run_input=run_input,
            build=self.build,
            memory_mbytes=self.memory_mbytes,
            run_timeout=to_timedelta(self.timeout_secs),
            max_total_charge_usd=to_decimal(self.max_total_charge_usd),
        )
        dataset_id = get_dataset_id(run, f'Actor {self.actor_id}')
        return {'documents': _load_documents(client, dataset_id, self.dataset_mapping_function)}

    @component.output_types(documents=list[Document])
    async def run_async(self, run_input: dict | None = None) -> dict[str, list[Document]]:
        """Run an Actor on the Apify platform and wait for results to be ready, asynchronously.

        Args:
            run_input (dict, optional): The input parameters for the Actor you want to run. This can be provided
                either in the `run_async` method or during the class instantiation. If specified in both places,
                the inputs are merged and the values from `run_async` take precedence. Without any input, the
                Actor runs with its default input.
        """
        run_input = _merge_inputs(self.run_input, run_input)

        client = self._get_async_client()
        run = await client.actor(self.actor_id).call(
            run_input=run_input,
            build=self.build,
            memory_mbytes=self.memory_mbytes,
            run_timeout=to_timedelta(self.timeout_secs),
            max_total_charge_usd=to_decimal(self.max_total_charge_usd),
        )
        dataset_id = get_dataset_id(run, f'Actor {self.actor_id}')
        documents = await _load_documents_async(client, dataset_id, self.dataset_mapping_function)
        return {'documents': documents}

    def to_dict(self) -> dict[str, Any]:
        """Serialize the component to a dictionary."""
        return default_to_dict(
            self,
            actor_id=self.actor_id,
            dataset_mapping_function=_serialize_mapping_function(self.dataset_mapping_function),
            run_input=self.run_input,
            build=self.build,
            memory_mbytes=self.memory_mbytes,
            timeout_secs=self.timeout_secs,
            max_total_charge_usd=self.max_total_charge_usd,
            apify_token=self.apify_token,
        )


@component
class ApifyDatasetFromTaskCall(ApifyDatasetComponent):
    """Get an Apify dataset by calling an Apify task.

    Perform a call of an Apify task and load the produced dataset.
    For details, see https://docs.apify.com/platform/integrations/haystack

    It needs an Apify API token, read from the `APIFY_TOKEN` environment variable by default, or passed as
    `apify_token`.
    """

    # Serialized pipelines store the class by this public path, so it does not depend on the module layout.
    __module__ = 'apify_haystack'

    def __init__(
        self,
        task_id: str,
        dataset_mapping_function: Callable[[dict], Document] | None = None,
        task_input: dict | None = None,
        *,
        build: str | None = None,
        memory_mbytes: int | None = None,
        timeout_secs: int | None = None,
        max_total_charge_usd: float | None = None,
        apify_token: Secret = Secret.from_env_var(['APIFY_TOKEN', 'APIFY_API_TOKEN']),
    ) -> None:
        """Initialize the Apify task loader.

        Args:
            task_id (str): The ID or name of the task on the Apify platform.
            dataset_mapping_function (Callable, optional): A function that takes a single dictionary (an Apify dataset
                item) and converts it to an instance of the Document class. Defaults to
                `default_dataset_mapping_function`.
            task_input (dict, optional): The input parameters for the task you want to run. This can be provided
                either in the `run` method or during the class instantiation. If specified in both places,
                the inputs are merged and the values from `run` take precedence. Without any input, the
                task runs with its saved input.
            build (str, optional): Optionally specifies the Actor build to run.
                It can be either a build tag or build number.
            memory_mbytes (int, optional): Optional memory limit for the run, in megabytes.
            timeout_secs (int, optional): Optional timeout for the run, in seconds.
            max_total_charge_usd (float, optional): The maximum amount in USD to charge for the run of a pay-per-event
                Actor. The platform charges no more than that.
            apify_token (Secret, optional): The API token for the Apify platform. Defaults to the
                `APIFY_TOKEN` environment variable, or the older `APIFY_API_TOKEN`.
        """
        self.task_id = task_id
        self.dataset_mapping_function = dataset_mapping_function or default_dataset_mapping_function
        self.task_input = task_input
        self.build = build
        self.memory_mbytes = memory_mbytes
        self.timeout_secs = timeout_secs
        self.max_total_charge_usd = max_total_charge_usd
        self.apify_token = apify_token
        self.client = None
        self.async_client = None

    @component.output_types(documents=list[Document])
    def run(self, task_input: dict | None = None) -> dict[str, list[Document]]:
        """Run a task on the Apify platform and wait for results to be ready.

        Args:
            task_input (dict, optional): The input parameters for the task you want to run. This can be provided
                either in the `run` method or during the class instantiation. If specified in both places,
                the inputs are merged and the values from `run` take precedence. Without any input, the
                task runs with its saved input.
        """
        task_input = _merge_inputs(self.task_input, task_input)

        client = self._get_client()
        run = client.task(self.task_id).call(
            task_input=task_input,
            build=self.build,
            memory_mbytes=self.memory_mbytes,
            run_timeout=to_timedelta(self.timeout_secs),
            max_total_charge_usd=to_decimal(self.max_total_charge_usd),
        )
        dataset_id = get_dataset_id(run, f'task {self.task_id}')
        return {'documents': _load_documents(client, dataset_id, self.dataset_mapping_function)}

    @component.output_types(documents=list[Document])
    async def run_async(self, task_input: dict | None = None) -> dict[str, list[Document]]:
        """Run a task on the Apify platform and wait for results to be ready, asynchronously.

        Args:
            task_input (dict, optional): The input parameters for the task you want to run. This can be provided
                either in the `run_async` method or during the class instantiation. If specified in both places,
                the inputs are merged and the values from `run_async` take precedence. Without any input, the
                task runs with its saved input.
        """
        task_input = _merge_inputs(self.task_input, task_input)

        client = self._get_async_client()
        run = await client.task(self.task_id).call(
            task_input=task_input,
            build=self.build,
            memory_mbytes=self.memory_mbytes,
            run_timeout=to_timedelta(self.timeout_secs),
            max_total_charge_usd=to_decimal(self.max_total_charge_usd),
        )
        dataset_id = get_dataset_id(run, f'task {self.task_id}')
        documents = await _load_documents_async(client, dataset_id, self.dataset_mapping_function)
        return {'documents': documents}

    def to_dict(self) -> dict[str, Any]:
        """Serialize the component to a dictionary."""
        return default_to_dict(
            self,
            task_id=self.task_id,
            dataset_mapping_function=_serialize_mapping_function(self.dataset_mapping_function),
            task_input=self.task_input,
            build=self.build,
            memory_mbytes=self.memory_mbytes,
            timeout_secs=self.timeout_secs,
            max_total_charge_usd=self.max_total_charge_usd,
            apify_token=self.apify_token,
        )


def default_dataset_mapping_function(dataset_item: dict) -> Document:
    """Convert an Apify dataset item to a `Document`.

    It fits the items of the website crawling and web search Actors, such as Website Content Crawler and RAG Web
    Browser: the content is the item's `markdown`, or its `text`, and `meta` holds the page `url`, `title`, and
    `description`. An item with neither `markdown` nor `text` becomes a `Document` with the whole item as JSON.
    """
    # `metadata` describes a scraped page, `searchResult` the search result it came from.
    page = next((value for value in (dataset_item.get('metadata'), dataset_item.get('searchResult')) if value), {})
    if not isinstance(page, dict):
        page = {}

    content = dataset_item.get('markdown') or dataset_item.get('text')
    if not isinstance(content, str) or not content:
        content = json.dumps(dataset_item, ensure_ascii=False, default=str)

    meta = {
        'url': dataset_item.get('url') or page.get('url'),
        'title': page.get('title'),
        'description': page.get('description'),
    }
    return Document(content=content, meta={key: value for key, value in meta.items() if value is not None})


def _serialize_mapping_function(dataset_mapping_function: Callable[[dict], Document]) -> str | None:
    # The default is stored as `None`, so loading a pipeline that uses it needs no allowlisted module.
    if dataset_mapping_function is default_dataset_mapping_function:
        return None
    return serialize_callable(dataset_mapping_function)


def _merge_inputs(input1: dict | None, input2: dict | None) -> dict | None:
    merged = (input1 or {}) | (input2 or {})
    return merged or None


def _load_documents(
    client: ApifyClient,
    dataset_id: str,
    dataset_mapping_function: Callable[[dict], Document],
) -> list[Document]:
    dataset_items = client.dataset(dataset_id).list_items(clean=True).items
    return [dataset_mapping_function(item) for item in dataset_items]


async def _load_documents_async(
    client: ApifyClientAsync,
    dataset_id: str,
    dataset_mapping_function: Callable[[dict], Document],
) -> list[Document]:
    dataset_items = (await client.dataset(dataset_id).list_items(clean=True)).items
    return [dataset_mapping_function(item) for item in dataset_items]
