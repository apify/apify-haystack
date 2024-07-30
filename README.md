# Apify-Haystack integration

The Apify-Haystack integration allows easy interaction between the Apify platform and Haystack.

Apify is a platform for web scraping, data extraction, and web automation tasks.
It provides serverless applications called Actors for different tasks, like crawling websites, and scraping Facebook, Instagram, and Google results, etc.

Haystack offers an ecosystem of tools for building, managing, and deploying search engines and LLM applications.

## Installation

Apify-haystack is available at the [`apify-haystack`](https://pypi.org/project/apify-haystack/) PyPI package.

```sh
pip install apify-haystack
```

## Examples

Crawl website using Apify Website Content Crawler and convert it to Haystack Documents.

You need to have an Apify account and API token to run this example.
You can start with a free account at [Apify](https://apify.com/) and get your [API token](https://docs.apify.com/platform/integrations/api).

In the example below, specify `apify_api_token` and run the script:

```python
from dotenv import load_dotenv
from haystack import Document

from apify_haystack import ApifyDatasetFromActorCall

# Sey APIFY-API-TOKEN here or load it from .env file
apify_api_token = "" or load_dotenv()

actor_id = "apify/website-content-crawler"
run_input = {
    "maxCrawlPages": 3,  # limit the number of pages to crawl
    "startUrls": [{"url": "https://haystack.deepset.ai/"}],
}


def dataset_mapping_function(dataset_item: dict) -> Document:
    return Document(content=dataset_item.get("text"), meta={"url": dataset_item.get("url")})


actor = ApifyDatasetFromActorCall(
    actor_id=actor_id, run_input=run_input, dataset_mapping_function=dataset_mapping_function
)
print(f"Calling the Apify actor {actor_id} ... crawling will take some time ...")
print("You can monitor the progress at: https://console.apify.com/actors/runs")

dataset = actor.run().get("documents")

print(f"Loaded {len(dataset)} documents from the Apify Actor {actor_id}:")
for d in dataset:
    print(d)
```

See other examples in the [examples directory](https://github.com/apify/apify-haystack/blob/master/src/apify_haystack/examples) for more examples, here is a list of few of them

- Load a dataset from Apify and convert it to Haystack Document
- Call [Website Content Crawler](https://apify.com/apify/website-content-crawler) Actor and convert the data into the Haystack Documents
- Crawl website, retrieve text content, and store it in the `InMemoryDocumentStore`
- Retrieval-Augmented Generation (RAG): Extracting text from a website & question answering

## Support

If you find any bug or issue, please [submit an issue on GitHub](https://github.com/apify/apify-haystack/issues).
For questions, you can ask on [Stack Overflow](https://stackoverflow.com/questions/tagged/apify), in GitHub Discussions or you can join our [Discord server](https://discord.com/invite/jyEM2PRvMU).

## Contributing

Your code contributions are welcome.
If you have any ideas for improvements, either submit an issue or create a pull request.
For contribution guidelines and the code of conduct, see [CONTRIBUTING.md](https://github.com/apify/apify-haystack/blob/master/CONTRIBUTING.md).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/apify/apify-haystack/blob/master/LICENSE) file for details.
