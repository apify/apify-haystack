from typing import Callable, Dict, List

from apify_client import ApifyClient
from constants import HAYSTACK_ATTRIBUTE_USER_AGENT
from haystack import component
from haystack.dataclasses import Document


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

    def __init__(self, dataset_id: str, dataset_mapping_function: Callable[[Dict], Document]):

        self.dataset_id = dataset_id
        self.dataset_mapping_function = dataset_mapping_function
        self.client = ApifyClient()

        # Add a custom user-agent to the httpx client for attribution purposes
        self.client = ApifyClient()
        if httpx_client := self.client.http_client.httpx_client:
            httpx_client.headers["user-agent"] += f"; {HAYSTACK_ATTRIBUTE_USER_AGENT}"

    @component.output_types(documents=List[Document])
    def run(self):
        """Load datasets produced by the `Apify Actors`."""
        dataset_items = self.client.dataset(self.dataset_id).list_items(clean=True).items
        return list(map(self.dataset_mapping_function, dataset_items))
