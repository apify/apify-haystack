# Apify-Haystack integration

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/apify/apify-haystack/blob/main/LICENSE)
[![PyPi Package](https://badge.fury.io/py/apify-haystack.svg)](https://badge.fury.io/py/apify-haystack)
[![Python](https://img.shields.io/pypi/pyversions/apify-haystack)](https://pypi.org/project/apify-haystack)

The Apify-Haystack integration allows easy interaction between the [Apify](https://apify.com/) platform and [Haystack](https://haystack.deepset.ai/).

Apify is a platform for web scraping, data extraction, and web automation tasks.
It provides serverless applications called Actors for tasks like crawling websites and scraping Facebook, Instagram, and Google results.

Haystack is an open-source framework for building LLM applications, RAG pipelines, and agents.

For a step-by-step guide, see the [Haystack integration](https://docs.apify.com/platform/integrations/haystack) page in the Apify documentation.

## Installation

The integration is available as the [`apify-haystack`](https://pypi.org/project/apify-haystack/) package on PyPI. It requires Python 3.11 or newer and Haystack 3.

```sh
pip install apify-haystack
```

## Components

- `ApifyDatasetFromActorCall` runs an Actor, waits for it to finish, and loads its default dataset.
- `ApifyDatasetFromTaskCall` does the same for a saved Actor task.
- `ApifyDatasetLoader` loads an existing dataset by its ID.
- `ApifyWebSearch` searches the web with the [RAG Web Browser](https://apify.com/apify/rag-web-browser) Actor and returns the content of the top result pages.
- `ApifyActorTool` is a tool for Haystack Agents that runs any Actor and returns its dataset items. It takes its parameters from the Actor's input schema, and it marks the items it hands to the LLM as untrusted external content to guard against prompt injection.

The dataset components turn dataset items into Haystack `Document` objects with a mapping function. Without one, they use `default_dataset_mapping_function`. For website and search Actors, it takes the item's `markdown` or `text` as the content and the page `url`, `title`, and `description` as `meta`. For other Actors, the content is the whole item as JSON.

Every component also has `run_async`, which `Pipeline.run_async` uses to run it without blocking the event loop.

When an Actor or task run finishes with any status other than `SUCCEEDED`, the component raises a `RuntimeError`.

The components read your Apify API token from the `APIFY_TOKEN` environment variable, or from `APIFY_API_TOKEN` if `APIFY_TOKEN` isn't set. To pass the token differently, use a Haystack `Secret`, for example `apify_token=Secret.from_env_var('MY_APIFY_TOKEN')`.

## Examples

The examples need an Apify account and API token. You can start with a free account at [Apify](https://apify.com/) and get your [API token](https://docs.apify.com/platform/integrations/api), then set it as the `APIFY_TOKEN` environment variable.

### Crawl a website and load it as Documents

`ApifyDatasetFromActorCall` runs [Website Content Crawler](https://apify.com/apify/website-content-crawler) and loads the crawled pages as `Document` objects with the default mapping function:

```python
from apify_haystack import ApifyDatasetFromActorCall

crawler = ApifyDatasetFromActorCall(
    actor_id='apify/website-content-crawler',
    run_input={
        'maxCrawlPages': 3,
        'startUrls': [{'url': 'https://haystack.deepset.ai/'}],
    },
)

for document in crawler.run()['documents']:
    print(document.meta['url'], document.content[:100])
```

### Search the web

`ApifyWebSearch` returns the top result pages as Markdown `Document` objects, with the page `url`, `title`, and `description` in `meta`, and their URLs in `links`:

```python
from apify_haystack import ApifyWebSearch

search = ApifyWebSearch(top_k=3)
result = search.run(query='What is Haystack?')

for document in result['documents']:
    print(document.meta['url'], document.meta['title'])
```

### Give an Agent access to an Actor

`ApifyActorTool` lets a Haystack `Agent` run an Actor. Limit `dataset_items_limit` to keep the tool output within the model's context window. For pay-per-event Actors, `max_total_charge_usd` caps what each run the Agent starts can cost:

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

from apify_haystack import ApifyActorTool

tool = ApifyActorTool('apify/rag-web-browser', dataset_items_limit=3, max_total_charge_usd=0.5)
agent = Agent(chat_generator=OpenAIChatGenerator(), tools=[tool])

result = agent.run([ChatMessage.from_user('What is new in Haystack 3?')])
print(result['last_message'].text)
```

### Save and load a pipeline

The components and the tool support Haystack serialization. The token is saved as the name of its environment variable, never as its value. Haystack 3 only loads classes and functions from trusted modules, so pass `apify_haystack` in `allowed_modules`, along with the module of your mapping function if you use a custom one:

```python
from haystack import Pipeline

from apify_haystack import ApifyDatasetLoader

pipeline = Pipeline()
pipeline.add_component('loader', ApifyDatasetLoader('YOUR-DATASET-ID'))

yaml = pipeline.dumps()
restored = Pipeline.loads(yaml, allowed_modules=['apify_haystack'])
```

### More examples

The [examples directory](https://github.com/apify/apify-haystack/tree/main/examples) has complete scripts:

- [Load an existing dataset](https://github.com/apify/apify-haystack/blob/main/examples/apify_dataset_load.py) and convert its items to Documents.
- [Call Website Content Crawler](https://github.com/apify/apify-haystack/blob/main/examples/apify_actor_call.py) with a custom mapping function.
- [Crawl a website, store the pages in `InMemoryDocumentStore`, and retrieve them](https://github.com/apify/apify-haystack/blob/main/examples/crawl_and_process_data.py).
- [Answer questions about a crawled website with RAG](https://github.com/apify/apify-haystack/blob/main/examples/rag_with_crawled_website.py). <a href="https://colab.research.google.com/github/deepset-ai/haystack-cookbook/blob/main/notebooks/apify_haystack_rag.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- [Analyze the sentiment of Instagram comments](https://github.com/apify/apify-haystack/blob/main/examples/instagram_comments_analysis.py). <a href="https://colab.research.google.com/github/deepset-ai/haystack-cookbook/blob/main/notebooks/apify_haystack_instagram_comments_analysis.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Support

If you find a bug, please [submit an issue on GitHub](https://github.com/apify/apify-haystack/issues).
For questions, ask on [Stack Overflow](https://stackoverflow.com/questions/tagged/apify) or join our [Discord server](https://discord.com/invite/jyEM2PRvMU).

## Contributing

Code contributions are welcome.
If you have ideas for improvements, submit an issue or create a pull request.
For contribution guidelines, see [CONTRIBUTING.md](https://github.com/apify/apify-haystack/blob/main/CONTRIBUTING.md).

## License

This project is licensed under the [Apache License 2.0](https://github.com/apify/apify-haystack/blob/main/LICENSE).
