from __future__ import annotations

import os
from typing import Callable

from apify_client import ApifyClient
from haystack import component
from haystack.dataclasses import Document

from apify_haystack.constants import HAYSTACK_ATTRIBUTE_USER_AGENT


@component
class ApifyDatasetLoader:
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

        # Add a custom user-agent to the httpx client for attribution purposes
        self.client = ApifyClient()
        if httpx_client := getattr(self.client.http_client, "httpx_client", None):
            httpx_client.headers["user-agent"] += f"; {HAYSTACK_ATTRIBUTE_USER_AGENT}"

    @component.output_types(documents=list[Document])
    def run(self):  # type: ignore[no-untyped-def] # noqa: ANN201
        """Load datasets produced by the `Apify Actors`.

        Type-hint note: the output is not type-hinted, otherwise linting complains
        that `run` method is not haystack a component
        """
        dataset_items = self.client.dataset(self.dataset_id).list_items(clean=True).items
        return {"documents": list(map(self.dataset_mapping_function, dataset_items))}


@component
class ApifyDatasetFromActorCall:
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
                the inputs will be merged.
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

        if not (apify_api_token := apify_api_token or os.getenv("APIFY_API_TOKEN")):
            raise ValueError(
                "Apify API token not found. Please provide Apify API token when initializing "
                "ApifyDatasetCall, or set the environment variable APIFY_API_TOKEN."
            )
        # Add a custom user-agent to the httpx client for attribution purposes
        self.client = ApifyClient(apify_api_token)

        if httpx_client := getattr(self.client.http_client, "httpx_client", None):
            httpx_client.headers["user-agent"] += f"; {HAYSTACK_ATTRIBUTE_USER_AGENT}"

    @component.output_types(documents=list[Document])  # type: ignore[misc]
    def run(self, run_input: dict | None = None):  # type: ignore[no-untyped-def] # noqa: ANN201
        """Run an Actor on the Apify platform and wait for results to be ready.

        Args:
            run_input (dict, optional): The input parameters for the Actor you want to run. This can be provided
                either in the `run` method or during the class instantiation. If specified in both places,
                the inputs will be merged.

        Example:
            .. code-block:: python

                actor = ApifyDatasetFromActorCall(
                    actor_id="apify/website-content-crawler",
                    run_input={ "maxCrawlPages": 5 }
                )
                dataset = actor.run(run_input={ "startUrls": [{"url": "https://haystack.deepset.ai/"}] })

        Type-hint note: the output is not type-hinted, otherwise linting complains
        that `run` method is not haystack a component
        """
        if not (run_input := _merge_inputs(run_input, self.run_input)):
            raise ValueError("Please provide the run_input either in the constructor or in the run method.")

        if not (
            actor_call := self.client.actor(self.actor_id).call(
                run_input=run_input,
                build=self.build,
                memory_mbytes=self.memory_mbytes,
                timeout_secs=self.timeout_secs,
            )
        ):
            raise ValueError(f"No response found in the actor {self.actor_id} call")

        if not (dataset_id := actor_call.get("defaultDatasetId")):
            raise ValueError(f"No dataset found in the actor {self.actor_id} call response.")

        loader = ApifyDatasetLoader(
            dataset_id=dataset_id,
            dataset_mapping_function=self.dataset_mapping_function,
        )
        return loader.run()


@component
class ApifyDatasetFromTaskCall:
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
                the inputs will be merged.
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

        if not (apify_api_token := apify_api_token or os.getenv("APIFY_API_TOKEN")):
            raise ValueError(
                "Apify API token not found. Please provide Apify API token when initializing "
                "ApifyDatasetCall, or set the environment variable APIFY_API_TOKEN."
            )
        # Add a custom user-agent to the httpx client for attribution purposes
        self.client = ApifyClient(apify_api_token)

        if httpx_client := getattr(self.client.http_client, "httpx_client", None):
            httpx_client.headers["user-agent"] += f"; {HAYSTACK_ATTRIBUTE_USER_AGENT}"

    @component.output_types(documents=list[Document])  # type: ignore[misc]
    def run(self, task_input: dict | None = None):  # type: ignore[no-untyped-def] # noqa: ANN201
        """Run an Actor on the Apify platform and wait for results to be ready.

        Args:
            task_input (dict, optional): The input parameters for the Actor you want to run. This can be provided
                either in the `run` method or during the class instantiation. If specified in both places,
                the inputs will be merged.

        Type-hint note: the output is not type-hinted, otherwise linting complains
        that `run` method is not haystack a component
        """
        task_input = task_input or self.task_input
        if not task_input:
            raise ValueError("Please provide the task_input either in the constructor or in the run method.")

        if not (
            actor_call := self.client.task(self.task_id).call(
                task_input=self.task_input,
                build=self.build,
                memory_mbytes=self.memory_mbytes,
                timeout_secs=self.timeout_secs,
            )
        ):
            raise ValueError(f"No response found in the actor {self.task_id} call")

        if not (dataset_id := actor_call.get("defaultDatasetId")):
            raise ValueError(f"No dataset found in the actor {self.task_id} call response.")

        loader = ApifyDatasetLoader(
            dataset_id=dataset_id,
            dataset_mapping_function=self.dataset_mapping_function,
        )
        return loader.run()


def _merge_inputs(input1: dict | None, input2: dict | None) -> dict | None:
    if input1 and input2:
        return input1 | input2
    return input1 or input2
