"""Call Apify Actor and Load a dataset

Let's say that you want to crawl and get text  content from you https://docs.haystack.deepset.ai/
You can call Apify's Website Content Crawler and get the scraped data in a dataset.

The script should produce the following output (an example of a single Document):
......
Document(id=a617d376*****, content: 'Introduction to Haystack 2.x)
Haystack is an open-source framework fo...', meta: 'url': 'https://docs.haystack.deepset.ai/docs/intro'
.....
"""

from dotenv import load_dotenv
from haystack import Document

from apify_dataset import ApifyDatasetFromActorCall

# Sey APIFY-API-TOKEN here or load it from .env file
apify_token = "" or load_dotenv()

actor_id = "apify/website-content-crawler"
run_input = {
    "maxCrawlPages": 3,  # limit the number of pages to crawl
    "startUrls": [{"url": "https://docs.haystack.deepset.ai/"}],
}


def dataset_mapping_function(dataset_item: dict) -> Document:
    return Document(content=dataset_item.get("text"), meta={"url": dataset_item.get("url")})


actor = ApifyDatasetFromActorCall(
    actor_id=actor_id, run_input=run_input, dataset_mapping_function=dataset_mapping_function
)
print(f"Calling Apify actor {actor_id} ... crawling will take some time.")
dataset = actor.run()

print(f"Loaded {len(dataset)} documents from the Apify Actor {actor_id}:")
for d in dataset:
    print(d)
