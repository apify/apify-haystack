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

See the [examples directory](https://github.com/apify/apify-haystack/blob/master/src/apify_haystack/examples) for more examples, here is a list of few of them

- Load a dataset from Apify and convert it to Haystack Documents
- Call Apify Actor and load a dataset to convert it to Haystack Documents
- Crawl website, scrape text content, and store it in the InMemoryDocumentStore
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
