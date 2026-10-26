from __future__ import annotations

from haystack import Document

from apify_haystack import ApifyWebSearch

# With a URL as the query, RAG Web Browser fetches that page instead of searching, which keeps the result fixed.
QUERY = 'https://crawlee.dev'

# Raw HTTP is the fastest scraping tool, and the Crawlee homepage renders without a browser.
SEARCH_PARAMS = {'scrapingTool': 'raw-http'}


def test_search_returns_page_as_document() -> None:
    """A search returns the page content as Markdown with its URL in `meta` and `links`, within the charge limit."""
    search = ApifyWebSearch(top_k=1, search_params=SEARCH_PARAMS, max_total_charge_usd=0.1)

    result = search.run(query=QUERY)
    search.close()

    [document] = result['documents']
    assert isinstance(document, Document)
    assert document.content is not None
    assert 'Crawlee' in document.content
    assert document.meta['url'].startswith(QUERY)
    assert result['links'] == [document.meta['url']]


async def test_search_run_async_returns_page_as_document() -> None:
    """`run_async` returns the same result through the async client."""
    search = ApifyWebSearch(top_k=1, search_params=SEARCH_PARAMS, max_total_charge_usd=0.1)

    result = await search.run_async(query=QUERY)
    await search.close_async()

    [document] = result['documents']
    assert isinstance(document, Document)
    assert document.content is not None
    assert 'Crawlee' in document.content
    assert result['links'] == [document.meta['url']]
