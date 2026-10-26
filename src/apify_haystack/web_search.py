from __future__ import annotations

from typing import Any, Self

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.utils import Secret

from apify_haystack.apify_dataset import _ApifyClientComponent, _to_timedelta

RAG_WEB_BROWSER_ACTOR_ID = 'apify/rag-web-browser'

# The RAG Web Browser Actor rejects a larger `maxResults`.
_MAX_TOP_K = 100


@component
class ApifyWebSearch(_ApifyClientComponent):
    """Search the web with the Apify RAG Web Browser Actor and return the result pages as Documents.

    The Actor searches Google, scrapes the top results, and returns their content as Markdown. Each result becomes
    a `Document` with the page content and the page `url`, `title`, and `description` in `meta`.
    For details about the Actor, see https://apify.com/apify/rag-web-browser

    Example:
        .. code-block:: python

            from apify_haystack import ApifyWebSearch

            search = ApifyWebSearch(top_k=3)
            result = search.run(query='What is Haystack?')
            documents, links = result['documents'], result['links']
    """

    def __init__(
        self,
        top_k: int = 5,
        search_params: dict | None = None,
        *,
        timeout_secs: int | None = None,
        apify_token: Secret = Secret.from_env_var(['APIFY_TOKEN', 'APIFY_API_TOKEN']),
    ) -> None:
        """Initialize the web search component.

        Args:
            top_k (int, optional): The number of result pages to return, from 1 to 100.
            search_params (dict, optional): Additional input for the RAG Web Browser Actor, for example
                `{'scrapingTool': 'browser-playwright'}`. See the Actor's input schema for the options.
            timeout_secs (int, optional): Optional timeout for the Actor run, in seconds.
            apify_token (Secret, optional): The API token for the Apify platform. Defaults to the
                `APIFY_TOKEN` environment variable, or the older `APIFY_API_TOKEN`.
        """
        if not 1 <= top_k <= _MAX_TOP_K:
            raise ValueError(f'top_k must be between 1 and {_MAX_TOP_K}, got {top_k}.')

        self.top_k = top_k
        self.search_params = search_params
        self.timeout_secs = timeout_secs
        self.apify_token = apify_token
        self.client = None
        self.async_client = None

    @component.output_types(documents=list[Document], links=list[str])
    def run(self, query: str) -> dict[str, list[Document] | list[str]]:
        """Search the web and return the content of the top result pages.

        Args:
            query (str): The search query.
        """
        client = self._get_client()
        run = client.actor(RAG_WEB_BROWSER_ACTOR_ID).call(
            run_input=self._run_input(query),
            run_timeout=_to_timedelta(self.timeout_secs),
        )
        if run is None:
            raise ValueError(f'No response found in the Actor {RAG_WEB_BROWSER_ACTOR_ID} call')

        items = client.dataset(run.default_dataset_id).list_items(clean=True).items
        return _to_output(items)

    @component.output_types(documents=list[Document], links=list[str])
    async def run_async(self, query: str) -> dict[str, list[Document] | list[str]]:
        """Search the web and return the content of the top result pages, asynchronously.

        Args:
            query (str): The search query.
        """
        client = self._get_async_client()
        run = await client.actor(RAG_WEB_BROWSER_ACTOR_ID).call(
            run_input=self._run_input(query),
            run_timeout=_to_timedelta(self.timeout_secs),
        )
        if run is None:
            raise ValueError(f'No response found in the Actor {RAG_WEB_BROWSER_ACTOR_ID} call')

        items = (await client.dataset(run.default_dataset_id).list_items(clean=True)).items
        return _to_output(items)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the component to a dictionary."""
        return default_to_dict(
            self,
            top_k=self.top_k,
            search_params=self.search_params,
            timeout_secs=self.timeout_secs,
            apify_token=self.apify_token,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize the component from a dictionary created by `to_dict`."""
        return default_from_dict(cls, data)

    def _run_input(self, query: str) -> dict:
        return {**(self.search_params or {}), 'query': query, 'maxResults': self.top_k}


def _to_output(items: list[dict]) -> dict[str, list[Document] | list[str]]:
    documents = []
    for item in items:
        # `metadata` describes the scraped page, `searchResult` the Google result it came from.
        page = item.get('metadata') or item.get('searchResult') or {}
        documents.append(
            Document(
                content=item.get('markdown') or item.get('text') or '',
                meta={'url': page.get('url'), 'title': page.get('title'), 'description': page.get('description')},
            )
        )
    return {'documents': documents, 'links': [document.meta['url'] for document in documents if document.meta['url']]}
