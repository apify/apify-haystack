"""Load a dataset from Apify and convert it to a Haystack Document.

Let's assume that you have run an Apify actor that scraped web pages of https://docs.haystack.deepset.ai/
Having the dataset ID, you can load the dataset and convert it to Haystack Documents.

The script should produce the following output (an example of a single Document):
......
Document(id=a617d376*****, content: 'Introduction to Haystack 2.x)
Haystack is an open-source framework fo...', meta: 'url': 'https://docs.haystack.deepset.ai/docs/intro'
.....
"""

from haystack import Document

from apify_haystack import ApifyDatasetLoader

dataset_id = "YOUR-DATASET-ID"


def dataset_mapping_function(dataset_item: dict) -> Document:
    return Document(content=dataset_item.get("text"), meta={"url": dataset_item.get("url")})


loader = ApifyDatasetLoader(dataset_id=dataset_id, dataset_mapping_function=dataset_mapping_function)
documents = loader.run()
dataset = documents.get("documents")

print(f"Loaded {len(dataset)} documents from Apify dataset {dataset_id}:")
for d in dataset:
    print(d)
