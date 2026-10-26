from __future__ import annotations

import json
import re
from typing import Any

from apify_client import ApifyClient
from haystack.core.serialization import generate_qualified_class_name
from haystack.tools import Tool
from haystack.utils import Secret, deserialize_secrets_inplace

from apify_haystack._client import (
    ApifyClientComponent,
    create_client,
    get_dataset_id,
    to_decimal,
    to_timedelta,
)

# OpenAI limits a function name to 64 characters and a description to 1024.
_MAX_NAME_LENGTH = 64
_MAX_DESCRIPTION_LENGTH = 1000
_MAX_PROPERTY_DESCRIPTION_LENGTH = 350

# The markers around tool output that the LLM must treat as untrusted data, never as instructions.
UNTRUSTED_CONTENT_START = '<<<EXTERNAL_UNTRUSTED_CONTENT>>>'
UNTRUSTED_CONTENT_END = '<<<END_EXTERNAL_UNTRUSTED_CONTENT>>>'

# The keys of an Actor input schema property that a JSON schema for an LLM understands.
_PROPERTY_SCHEMA_KEYS = ('type', 'description', 'enum', 'default', 'items', 'minimum', 'maximum')


class ApifyActorTool(ApifyClientComponent, Tool):
    """A Haystack tool that runs an Apify Actor and returns the items of its default dataset.

    Use it to give a Haystack `Agent` access to any Actor from Apify Store. The tool takes its parameters from the
    Actor's input schema and its description from the Actor's definition, so the LLM sees the same inputs as a user
    of the Actor. Reading them needs one Apify API request when the tool is created without `parameters`.

    The LLM gets the dataset items as JSON between the `<<<EXTERNAL_UNTRUSTED_CONTENT>>>` and
    `<<<END_EXTERNAL_UNTRUSTED_CONTENT>>>` markers, since scraped content can carry prompt injections. `invoke` and
    `invoke_async` return the items unwrapped.
    For details, see https://docs.apify.com/platform/integrations/haystack

    Example:
        .. code-block:: python

            from haystack.components.agents import Agent
            from haystack.components.generators.chat import OpenAIChatGenerator

            from apify_haystack import ApifyActorTool

            tool = ApifyActorTool('apify/rag-web-browser', dataset_items_limit=3)
            agent = Agent(chat_generator=OpenAIChatGenerator(), tools=[tool])
    """

    # Serialized pipelines store the class by this public path, so it does not depend on the module layout.
    __module__ = 'apify_haystack'

    def __init__(
        self,
        actor_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        dataset_items_limit: int | None = None,
        build: str | None = None,
        memory_mbytes: int | None = None,
        timeout_secs: int | None = None,
        max_total_charge_usd: float | None = None,
        apify_token: Secret = Secret.from_env_var(['APIFY_TOKEN', 'APIFY_API_TOKEN']),
    ) -> None:
        """Initialize the Actor tool.

        Args:
            actor_id (str): The ID or name of the Actor on the Apify platform, for example `apify/rag-web-browser`.
            name (str, optional): The tool name the LLM sees. Defaults to `apify_actor_` followed by the Actor name.
            description (str, optional): The tool description the LLM sees. Defaults to the Actor's description.
            parameters (dict, optional): The JSON schema of the tool parameters. Defaults to a schema built from the
                Actor's input schema.
            dataset_items_limit (int, optional): The maximum number of dataset items to return. Limit it to keep the
                tool output within the LLM's context window. It only limits what the tool reads, not what the run
                produces or is charged for.
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
        self.dataset_items_limit = dataset_items_limit
        self.build = build
        self.memory_mbytes = memory_mbytes
        self.timeout_secs = timeout_secs
        self.max_total_charge_usd = max_total_charge_usd
        self.apify_token = apify_token
        self.client = None
        self.async_client = None

        if parameters is None or description is None:
            actor_definition = _get_actor_definition(apify_token, actor_id)
            if parameters is None:
                parameters = _to_tool_parameters(actor_definition.get('input') or {})
            if description is None:
                description = actor_definition.get('description') or f'Run the Apify Actor {actor_id}.'

        super().__init__(
            name=name or _to_tool_name(actor_id),
            description=description[:_MAX_DESCRIPTION_LENGTH],
            parameters=parameters,
            function=self._run_actor,
            async_function=self._run_actor_async,
            outputs_to_string={'handler': self._wrap_untrusted_content},
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the tool to a dictionary."""
        return {
            'type': generate_qualified_class_name(type(self)),
            'data': {
                'actor_id': self.actor_id,
                'name': self.name,
                'description': self.description,
                'parameters': self.parameters,
                'dataset_items_limit': self.dataset_items_limit,
                'build': self.build,
                'memory_mbytes': self.memory_mbytes,
                'timeout_secs': self.timeout_secs,
                'max_total_charge_usd': self.max_total_charge_usd,
                'apify_token': self.apify_token.to_dict(),
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ApifyActorTool:
        """Deserialize the tool from a dictionary created by `to_dict`. It makes no Apify API request."""
        init_parameters = dict(data['data'])
        deserialize_secrets_inplace(init_parameters, keys=['apify_token'])
        return cls(**init_parameters)

    def _wrap_untrusted_content(self, dataset_items: list[dict]) -> str:
        """Serialize the dataset items for the LLM, wrapped in the untrusted content markers."""
        content = json.dumps(dataset_items, ensure_ascii=False, default=str)
        # Scraped text could contain the markers and close the wrapper early.
        for marker in (UNTRUSTED_CONTENT_START, UNTRUSTED_CONTENT_END):
            content = content.replace(marker, '[marker removed]')
        return f'{UNTRUSTED_CONTENT_START}\nSource: apify:{self.actor_id}\n{content}\n{UNTRUSTED_CONTENT_END}'

    def _run_actor(self, **run_input: Any) -> list[dict]:
        client = self._get_client()
        run = client.actor(self.actor_id).call(
            run_input=run_input,
            build=self.build,
            memory_mbytes=self.memory_mbytes,
            run_timeout=to_timedelta(self.timeout_secs),
            max_total_charge_usd=to_decimal(self.max_total_charge_usd),
        )
        dataset_id = get_dataset_id(run, f'Actor {self.actor_id}')
        return client.dataset(dataset_id).list_items(clean=True, limit=self.dataset_items_limit).items

    async def _run_actor_async(self, **run_input: Any) -> list[dict]:
        client = self._get_async_client()
        run = await client.actor(self.actor_id).call(
            run_input=run_input,
            build=self.build,
            memory_mbytes=self.memory_mbytes,
            run_timeout=to_timedelta(self.timeout_secs),
            max_total_charge_usd=to_decimal(self.max_total_charge_usd),
        )
        dataset_id = get_dataset_id(run, f'Actor {self.actor_id}')
        return (await client.dataset(dataset_id).list_items(clean=True, limit=self.dataset_items_limit)).items


def _get_actor_definition(apify_token: Secret, actor_id: str) -> dict[str, Any]:
    client = create_client(ApifyClient, apify_token.resolve_value())
    try:
        build = client.actor(actor_id).default_build().get()
    finally:
        client.http_client.close()
    if build is None:
        raise ValueError(f'The default build of the Actor {actor_id} was not found.')
    # `by_alias` keeps the API's camelCase keys, and the definition fields apify-client does not model, such as
    # `description`, survive as extras.
    return build.model_dump(by_alias=True, mode='json').get('actorDefinition') or {}


def _to_tool_parameters(input_schema: dict[str, Any]) -> dict[str, Any]:
    properties = {}
    for property_name, property_schema in input_schema.get('properties', {}).items():
        tool_property = {key: property_schema[key] for key in _PROPERTY_SCHEMA_KEYS if key in property_schema}
        if description := tool_property.get('description'):
            tool_property['description'] = description[:_MAX_PROPERTY_DESCRIPTION_LENGTH]
        # The prefill is the example value the Apify Console shows in the input form.
        if 'prefill' in property_schema:
            tool_property['examples'] = [property_schema['prefill']]
        properties[property_name] = tool_property

    # The platform fills in the default of a missing property, so the LLM does not have to provide it.
    required = [name for name in input_schema.get('required', []) if 'default' not in properties.get(name, {})]
    return {'type': 'object', 'properties': properties, 'required': required}


def _to_tool_name(actor_id: str) -> str:
    return ('apify_actor_' + re.sub(r'[^a-zA-Z0-9_-]', '_', actor_id))[:_MAX_NAME_LENGTH]
