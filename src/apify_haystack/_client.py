from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, TypeVar

from apify_client import ApifyClient, ApifyClientAsync

from apify_haystack._constants import HAYSTACK_ATTRIBUTE_USER_AGENT, HAYSTACK_INTEGRATION_PLATFORM

if TYPE_CHECKING:
    from apify_client._models import Run
    from haystack.utils import Secret

_ClientT = TypeVar('_ClientT', ApifyClient, ApifyClientAsync)


class _ApifyClientComponent:
    """Base for the components that call the Apify API.

    The sync client lives between `warm_up` and `close`, the async one between `warm_up_async` and `close_async`.
    """

    apify_token: Secret
    client: ApifyClient | None
    async_client: ApifyClientAsync | None

    def warm_up(self) -> None:
        """Create the Apify client. `run` does this itself when the component was not warmed up."""
        self._get_client()

    async def warm_up_async(self) -> None:
        """Create the async Apify client. `run_async` does this itself when the component was not warmed up."""
        self._get_async_client()

    def close(self) -> None:
        """Release the Apify client. The next `warm_up` or `run` creates a new one."""
        if self.client is not None:
            self.client.http_client.close()
            self.client = None

    async def close_async(self) -> None:
        """Release the async Apify client. The next `warm_up_async` or `run_async` creates a new one."""
        if self.async_client is not None:
            await self.async_client.http_client.aclose()
            self.async_client = None

    def _get_client(self) -> ApifyClient:
        if self.client is None:
            self.client = _create_client(ApifyClient, self.apify_token.resolve_value())
        return self.client

    def _get_async_client(self) -> ApifyClientAsync:
        if self.async_client is None:
            self.async_client = _create_client(ApifyClientAsync, self.apify_token.resolve_value())
        return self.async_client


def _create_client(client_class: type[_ClientT], token: str | None) -> _ClientT:
    """Create an Apify client that identifies its requests as coming from this integration."""
    client = client_class(token, headers={'x-apify-integration-platform': HAYSTACK_INTEGRATION_PLATFORM})

    # Older platform paths attribute requests by the user-agent marker. Passing it through `headers` would replace
    # apify-client's own user agent, so the header is extended in place.
    client.http_client._headers['User-Agent'] += f'; {HAYSTACK_ATTRIBUTE_USER_AGENT}'  # noqa: SLF001
    return client


def _get_dataset_id(run: Run | None, label: str) -> str:
    """Return the default dataset of a finished run, raising unless the run succeeded."""
    if run is None:
        raise ValueError(f'No response found in the {label} call')
    if run.status != 'SUCCEEDED':
        raise RuntimeError(f'The {label} run {run.id} finished with status {run.status}.')
    return run.default_dataset_id


def _to_timedelta(seconds: int | None) -> timedelta | None:
    return None if seconds is None else timedelta(seconds=seconds)


def _to_decimal(value: float | None) -> Decimal | None:
    # `str` keeps the amount as written; `Decimal(0.1)` would carry the binary float error.
    return None if value is None else Decimal(str(value))
