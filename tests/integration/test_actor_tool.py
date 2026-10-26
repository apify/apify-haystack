from __future__ import annotations

from typing import Any

from haystack import component
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage, ToolCall

from apify_haystack import ApifyActorTool

# Web Fetch is fast, costs a fraction of a cent per page, and has a small input schema with one required field.
WEB_FETCH = 'apify/web-fetch'

# Without `formats`, Web Fetch returns an HTML page as Markdown.
TOOL_ARGUMENTS = {'url': 'https://crawlee.dev'}


@component
class ScriptedChatGenerator:
    """Asks for one call of the first tool, then answers with the tool result it got back."""

    # The `Agent` passes its tools to the chat generator, which this one reads the tool name from.
    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools: Any = None) -> dict[str, list[ChatMessage]]:
        tool_result = messages[-1].tool_call_result
        if tool_result is None:
            tool_call = ToolCall(tool_name=tools[0].name, arguments=TOOL_ARGUMENTS)
            return {'replies': [ChatMessage.from_assistant(tool_calls=[tool_call])]}
        return {'replies': [ChatMessage.from_assistant(str(tool_result.result))]}


def test_tool_is_built_from_actor_input_schema() -> None:
    """The tool takes its name, description, and parameters from the Actor's default build."""
    tool = ApifyActorTool(WEB_FETCH)

    assert tool.name == 'apify_actor_apify_web-fetch'
    assert tool.description
    properties = tool.parameters['properties']
    assert properties['url']['type'] == 'string'
    assert properties['url']['examples'] == ['https://apify.com']
    assert 'markdown' in properties['formats']['items']['enum']
    assert (properties['wait']['minimum'], properties['wait']['maximum']) == (1, 60000)
    assert tool.parameters['required'] == ['url']


def test_invoke_returns_dataset_items() -> None:
    """`invoke` runs the Actor with the tool arguments as input and returns at most `dataset_items_limit` items."""
    tool = ApifyActorTool(WEB_FETCH, dataset_items_limit=1, max_total_charge_usd=0.1)

    items = tool.invoke(**TOOL_ARGUMENTS)
    tool.close()

    [item] = items
    assert 'Crawlee' in item['markdown']


def test_agent_gets_result_as_untrusted_content() -> None:
    """In an `Agent`, the LLM gets the dataset items wrapped as untrusted content from the Actor."""
    tool = ApifyActorTool(WEB_FETCH, dataset_items_limit=1, max_total_charge_usd=0.1)
    agent = Agent(chat_generator=ScriptedChatGenerator(), tools=[tool])

    result = agent.run([ChatMessage.from_user('What is on crawlee.dev?')])
    tool.close()

    text = result['last_message'].text
    assert text.startswith('<<<EXTERNAL_UNTRUSTED_CONTENT>>>\nSource: apify:apify/web-fetch\n')
    assert text.endswith('\n<<<END_EXTERNAL_UNTRUSTED_CONTENT>>>')
    assert 'Crawlee' in text


async def test_agent_run_async_invokes_tool_asynchronously() -> None:
    """`Agent.run_async` runs the Actor through the tool's async client."""
    tool = ApifyActorTool(WEB_FETCH, dataset_items_limit=1, max_total_charge_usd=0.1)
    agent = Agent(chat_generator=ScriptedChatGenerator(), tools=[tool])

    result = await agent.run_async([ChatMessage.from_user('What is on crawlee.dev?')])
    await tool.close_async()

    assert tool.client is None
    assert 'Crawlee' in result['last_message'].text
