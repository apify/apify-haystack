# Apify-Haystack integration

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/apify/apify-haystack/blob/main/LICENSE)
[![PyPi Package](https://badge.fury.io/py/apify-haystack.svg)](https://badge.fury.io/py/apify-haystack)
[![Python](https://img.shields.io/pypi/pyversions/apify-haystack)](https://pypi.org/project/apify-haystack)

The Apify-Haystack integration allows easy interaction between the [Apify](https://apify.com/) platform and [Haystack](https://haystack.deepset.ai/).

Apify is a platform for web scraping, data extraction, and web automation tasks.
It provides serverless applications called Actors for tasks like crawling websites and scraping Facebook, Instagram, and Google results.

Haystack offers an ecosystem of tools for building, managing, and deploying search engines and LLM applications.

## Installation

Apify-haystack is available as the [`apify-haystack`](https://pypi.org/project/apify-haystack/) package on PyPI. It requires Python 3.11 or newer and Haystack 3.

```sh
pip install apify-haystack
```

## Components

- `ApifyDatasetFromActorCall` runs an Actor, waits for it to finish, and loads its default dataset.
- `ApifyDatasetFromTaskCall` does the same for a saved Actor task.
- `ApifyDatasetLoader` loads an existing dataset by its ID.

Each component turns dataset items into Haystack `Document` objects with a mapping function you provide. Every component also has `run_async`, which `Pipeline.run_async` uses to run it without blocking the event loop.

The components read your Apify API token from the `APIFY_API_TOKEN` environment variable. To pass the token differently, use a Haystack `Secret`, for example `apify_api_token=Secret.from_env_var('MY_APIFY_TOKEN')`.

## Examples

### Crawl a website using Apify's Website Content Crawler and convert it to Haystack Documents

You need an Apify account and API token to run this example.
You can start with a free account at [Apify](https://apify.com/) and get your [API token](https://docs.apify.com/platform/integrations/api).

In the example below, set the `APIFY_API_TOKEN` environment variable and run the script:

```python
from haystack import Document

from apify_haystack import ApifyDatasetFromActorCall

actor_id = 'apify/website-content-crawler'
run_input = {
    'maxCrawlPages': 3,  # limit the number of pages to crawl
    'startUrls': [{'url': 'https://haystack.deepset.ai/'}],
}


def dataset_mapping_function(dataset_item: dict) -> Document:
    return Document(content=dataset_item.get('text'), meta={'url': dataset_item.get('url')})


actor = ApifyDatasetFromActorCall(
    actor_id=actor_id, run_input=run_input, dataset_mapping_function=dataset_mapping_function
)
print(f'Calling the Apify Actor {actor_id}. Crawling will take some time.')
print('You can monitor the progress at: https://console.apify.com/actors/runs')

dataset = actor.run()['documents']

print(f'Loaded {len(dataset)} documents from the Apify Actor {actor_id}:')
for d in dataset:
    print(d)
```

### Save and load a pipeline

The components support Haystack serialization. The token is saved as the name of its environment variable, never as its value. Haystack 3 only loads classes and functions from trusted modules, so allow `apify_haystack` and the module with your mapping function:

```python
from haystack import Pipeline

yaml = pipeline.dumps()
restored = Pipeline.loads(yaml, allowed_modules=['apify_haystack', 'my_project.mapping'])
```

### More examples

The [examples directory](https://github.com/apify/apify-haystack/tree/main/examples) has more examples:

- Load a dataset from Apify and convert it to a Haystack Document
- Call [Website Content Crawler](https://apify.com/apify/website-content-crawler) and convert the data into the Haystack Documents
- Crawl websites, retrieve text content, and store it in the `InMemoryDocumentStore`
- Retrieval-Augmented Generation (RAG): Extracting text from a website & question answering <a href="https://colab.research.google.com/github/deepset-ai/haystack-cookbook/blob/main/notebooks/apify_haystack_rag.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- Analyze your Instagram comments' vibe with Apify and Haystack <a href="https://colab.research.google.com/github/deepset-ai/haystack-cookbook/blob/main/notebooks/apify_haystack_instagram_comments_analysis.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Support

If you find any bug or issue, please [submit an issue on GitHub](https://github.com/apify/apify-haystack/issues).
For questions, you can ask on [Stack Overflow](https://stackoverflow.com/questions/tagged/apify), in GitHub Discussions or you can join our [Discord server](https://discord.com/invite/jyEM2PRvMU).

## Contributing

Code contributions are welcome.
If you have ideas for improvements, submit an issue or create a pull request.
For contribution guidelines, see [CONTRIBUTING.md](https://github.com/apify/apify-haystack/blob/main/CONTRIBUTING.md).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/apify/apify-haystack/blob/main/LICENSE) file for details.
