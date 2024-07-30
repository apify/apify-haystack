# Apify-Haystack integration

The Apify-Haystack integration allows easy interaction between the Apify platform and Haystack.

Apify is a platform for web scraping, data extraction, and web automation tasks.
It provides serverless applications called Actors for different tasks, like crawling websites, and scraping Facebook, Instagram, and Google results, etc.

Haystack offers an ecosystem of tools for building, managing, and deploying search engines and LLM applications.

## Installation

Apify-haystack is available as the [`apify-haystack`](https://pypi.org/project/apify-haystack/) PyPI package.

```sh
pip install apify-haystack
```

## Examples

- Load a dataset from Apify and convert it to Haystack Documents: [apify_dataset_load.py](src/apify_haystack/examples/apify_dataset_load.py)
- Call Apify Actor and load a dataset to convert it to Haystack Documents: [apify_actor_call.py](src/apify_haystack/examples/apify_actor_call.py)
- Crawl website, scrape text content, and store it in the InMemoryDocumentStore: [crawl_and_process_data.py](src/apify_haystack/examples/crawl_and_process_data.py)
- Retrieval-Augmented Generation (RAG): Extracting text content from a website and using it for question answering [rag_with_crawled_website.py](src/apify_haystack/examples/rag_with_crawled_website.py)

## Support

If you find any bug or issue, please [submit an issue on GitHub](https://github.com/apify/apify-haystack/issues).
For questions, you can ask on [Stack Overflow](https://stackoverflow.com/questions/tagged/apify), in GitHub Discussions or you can join our [Discord server](https://discord.com/invite/jyEM2PRvMU).

## Contributing

Your code contributions are welcome, and you'll be praised for eternity!
If you have any ideas for improvements, either submit an issue or create a pull request.
For contribution guidelines and the code of conduct, see [CONTRIBUTING.md](https://github.com/apify/apify-haystack/blob/master/CONTRIBUTING.md).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/apify/apify-haystack/blob/master/LICENSE) file for details.
