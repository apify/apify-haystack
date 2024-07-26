from __future__ import annotations

import os
from abc import ABC, abstractmethod
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

    def __init__(self, dataset_id: str, dataset_mapping_function: Callable[[dict], Document]) -> None:

        self.dataset_id = dataset_id
        self.dataset_mapping_function = dataset_mapping_function
        self.client = ApifyClient()

        # Add a custom user-agent to the httpx client for attribution purposes
        self.client = ApifyClient()
        if httpx_client := getattr(self.client.http_client, "httpx_client", None):
            httpx_client.headers["user-agent"] += f"; {HAYSTACK_ATTRIBUTE_USER_AGENT}"

    @component.output_types(documents=list[Document])  # type: ignore[misc]
    def run(self) -> list[Document]:
        """Load datasets produced by the `Apify Actors`."""
        dataset_items = self.client.dataset(self.dataset_id).list_items(clean=True).items
        return list(map(self.dataset_mapping_function, dataset_items))


@component
class ApifyDatasetBase(ABC):
    """Base class for wrapper around Apify Actors.

    Perform call of Apify actors or tasks, and load datasets produced by them.
    """

    def __init__(
        self,
        actor_or_task_id: str,
        dataset_mapping_function: Callable[[dict], Document],
        apify_api_token: str | None = None,
    ) -> None:
        if not (apify_api_token := apify_api_token or os.getenv("APIFY_API_TOKEN")):
            raise ValueError(
                "Apify API token not found. Please provide Apify API token when initializing "
                "ApifyDatasetCall, or set the environment variable APIFY_API_TOKEN."
            )

        self.actor_or_task_id = actor_or_task_id
        self.dataset_mapping_function = dataset_mapping_function
        self.client = ApifyClient(apify_api_token)

        if httpx_client := getattr(self.client.http_client, "httpx_client", None):
            httpx_client.headers["user-agent"] += f"; {HAYSTACK_ATTRIBUTE_USER_AGENT}"

    @abstractmethod
    def run(self) -> list[Document]:
        """Run an Actor on the Apify platform and wait for results to be ready."""


@component
class ApifyDatasetFromActorCall(ApifyDatasetBase):
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
        apify_api_token: str | None = None,
    ) -> None:
        """Initialize the loader with an Apify actor ID and a mapping function.

        Args:
            actor_id (str): The ID or name of the actor on the Apify platform.
            dataset_mapping_function (Callable): A function that takes a single (an Apify dataset item) and converts
                it to an instance of the Document class.
            apify_api_token (str, optional): The API token for the Apify platform. If not provided, it will be read
                from the environment
        """
        super().__init__(actor_id, dataset_mapping_function, apify_api_token)

    @component.output_types(documents=list[Document])  # type: ignore[misc]
    def run(
        self,
        actor_id: str,
        run_input: dict,
        dataset_mapping_function: Callable[[dict], Document],
        *,
        build: str | None = None,
        memory_mbytes: int | None = None,
        timeout_secs: int | None = None,
    ) -> list[Document]:
        """Run an Actor on the Apify platform and wait for results to be ready.

        Args:
            actor_id (str): The ID or name of the Actor on the Apify platform.
            run_input (Dict): The input object of the Actor that you're trying to run.
            dataset_mapping_function (Callable): A function that takes a single
                dictionary (an Apify dataset item) and converts it to an
                instance of the Document class.
            build (str, optional): Optionally specifies the actor build to run.
                It can be either a build tag or build number.
            memory_mbytes (int, optional): Optional memory limit for the run,
                in megabytes.
            timeout_secs (int, optional): Optional timeout for the run, in seconds.

        Returns:
            list[Document]: A list of documents produced by the actor.
        """
        if not (
            actor_call := self.client.actor(actor_id).call(
                run_input=run_input,
                build=build,
                memory_mbytes=memory_mbytes,
                timeout_secs=timeout_secs,
            )
        ):
            raise ValueError(f"No response found in the actor {actor_id} call")

        if not (dataset_id := actor_call.get("defaultDatasetId")):
            raise ValueError(f"No dataset found in the actor {actor_id} call response.")

        loader = ApifyDatasetLoader(
            dataset_id=dataset_id,
            dataset_mapping_function=dataset_mapping_function,
        )
        return loader.run()  # type: ignore[no-any-return]

    @component
    class ApifyDatasetFromTaskCall(ApifyDatasetBase):
        """Get Apify dataset by calling Apify Task.

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
            apify_api_token: str | None = None,
        ) -> None:
            super().__init__(actor_id, dataset_mapping_function, apify_api_token)

        @component.output_types(documents=list[Document])  # type: ignore[misc]
        def run(
            self,
            task_id: str,
            task_input: dict,
            dataset_mapping_function: Callable[[dict], Document],
            *,
            build: str | None = None,
            memory_mbytes: int | None = None,
            timeout_secs: int | None = None,
        ) -> list[Document]:
            """Run a saved Actor task on Apify and wait for results to be ready.

            Args:
                task_id (str): The ID or name of the task on the Apify platform.
                task_input (Dict): The input object of the task that you're trying to run.
                    Overrides the task's saved input.
                dataset_mapping_function (Callable): A function that takes a single
                    dictionary (an Apify dataset item) and converts it to an
                    instance of the Document class.
                build (str, optional): Optionally specifies the actor build to run.
                    It can be either a build tag or build number.
                memory_mbytes (int, optional): Optional memory limit for the run,
                    in megabytes.
                timeout_secs (int, optional): Optional timeout for the run, in seconds.

            Returns:
                ApifyDatasetLoader: A loader that will fetch the records from the
                    task run's default dataset.
            """
            if not (
                task_call := self.client.task(task_id).call(
                    task_input=task_input, build=build, memory_mbytes=memory_mbytes, timeout_secs=timeout_secs
                )
            ):
                raise ValueError(f"No response found in the task {task_id} call")

            if not (dataset_id := task_call.get("defaultDatasetId")):
                raise ValueError(f"No dataset found in the task {task_id} call response.")

            loader = ApifyDatasetLoader(dataset_id=dataset_id, dataset_mapping_function=dataset_mapping_function)
            return loader.run()  # type: ignore[no-any-return]
