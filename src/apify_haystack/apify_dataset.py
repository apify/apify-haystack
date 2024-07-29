from __future__ import annotations

import os
from typing import Callable

from apify_client import ApifyClient
from haystack import component
from haystack.dataclasses import Document

from apify_haystack.constants import HAYSTACK_ATTRIBUTE_USER_AGENT


@component
class ApifyDatasetLoader:
    """Load datasets produced by the `Apify Actors`.

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

    @component.output_types(documents=list[Document])  # type: ignore[misc]
    def run(
        self,
    ) -> list[Document]:
        """Load datasets produced by the `Apify Actors`."""
        dataset_items = self.client.dataset(self.dataset_id).list_items(clean=True).items
        return list(map(self.dataset_mapping_function, dataset_items))


# @component
# class ApifyKeyValueStoreLoader:


class ApifyDatasetFromActorCall:
    """Get Apify dataset by calling Apify Actor.

    Perform call of Apify actor and load produced dataset.
    For details, see https://docs.apify.com/platform/integrations/haystack

    To use, you should have the ``apify-client`` python package installed,
    and the environment variable ``APIFY_API_TOKEN`` set with your API key, or pass
    `apify_api_token` as a named parameter to the constructor.
    """

    def __init__(
        self,
        actor_id: str,
        dataset_mapping_function: Callable[[dict], Document],
        run_input: dict,
        *,
        build: str | None = None,
        memory_mbytes: int | None = None,
        timeout_secs: int | None = None,
        apify_api_token: str | None = None,
    ) -> None:
        """
        Initialize the Apify Actor loader.

        Args:
            actor_id (str): The ID or name of the Actor on the Apify platform.
            run_input (Dict): The input object of the Actor that you're trying to run.
            dataset_mapping_function (Callable): A function that takes a single
                dictionary (an Apify dataset item) and converts it to an
                instance of the Document class.
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
    def run(self) -> list[Document]:
        """Run an Actor on the Apify platform and wait for results to be ready."""
        if not (
            actor_call := self.client.actor(self.actor_id).call(
                run_input=self.run_input,
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
