"""
Call Apify Actor and load a dataset to convert it to Haystack Documents.

This script demonstrates how to use Apify's Website Content Crawler to scrape text content from a specified website
and convert the scraped data into Haystack Documents.

For example, if you want to crawl and get text content from https://haystack.deepset.ai/, you can use this script
to call the Apify actor and retrieve the data in a structured format.

The script should produce the following output (an example of a single Document):
......
Document(id=a617d376*****, content: 'Introduction to Haystack 2.x)
Haystack is an open-source framework fo...', meta: {'url': 'https://docs.haystack.deepset.ai/docs/intro'}
.....
"""

from dotenv import load_dotenv
from haystack import Document

from apify_haystack import ApifyDatasetFromActorCall

# Sey APIFY-API-TOKEN here or load it from .env file
apify_token = "" or load_dotenv()

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
print(f"Calling Apify actor {actor_id} ... crawling will take some time ...")
dataset = actor.run().get("documents")

print(f"Loaded {len(dataset)} documents from the Apify Actor {actor_id}:")
for d in dataset:
    print(d)
