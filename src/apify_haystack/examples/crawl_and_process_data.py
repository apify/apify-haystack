"""
Crawl websites, scrape text content, and store it in the InMemoryDocumentStore.

This script demonstrates how to extract content from a website using Apify's Website Content Crawler.
The content is then cleaned, split into smaller chunks, embedded, and stored in the InMemoryDocumentStore.

After the pipeline is executed, the documents are retrieved from the document store
using BM25 retrieval and vector similarity.

The script should produce the following output (an example of a single Document):
......
Document(id=15a9f391a9f14871675826af8767bd3d2c81d7de1892fd9288b6be0dac26ea89, content: 'Get Started | Haystack
Haystack is an open-source Python framework that helps developers build LLM-p...',
meta: {'url': 'https://haystack.deepset.ai/overview/quick-start',
'source_id': '2a7263cb668b8353749cc990f487a3feb7a7d7f3abe3e3af3a69772cdecc8200', 'page_number': 1,
 'split_id': 0, 'split_idx_start': 0, '_split_overlap': ......
.....
"""

import os

from dotenv import load_dotenv
from haystack import Document, Pipeline
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils.auth import Secret

from apify_haystack import ApifyDatasetFromActorCall

load_dotenv()

# Set APIFY_API_TOKEN here or use it from .env file
apify_api_token = "" or os.getenv("APIFY_API_TOKEN")
openai_api_key = "" or os.getenv("OPENAI_API_KEY")

actor_id = "apify/website-content-crawler"
run_input = {
    "maxCrawlPages": 3,  # limit the number of pages to crawl
    "startUrls": [{"url": "https://haystack.deepset.ai/"}],
}


def dataset_mapping_function(dataset_item: dict) -> Document:
    return Document(content=dataset_item.get("text"), meta={"url": dataset_item.get("url")})


document_loader = ApifyDatasetFromActorCall(
    actor_id=actor_id,
    run_input=run_input,
    dataset_mapping_function=dataset_mapping_function,
    apify_api_token=apify_api_token,
)

document_store = InMemoryDocumentStore()
print(f"Initialized InMemoryDocumentStore with {document_store.count_documents()} documents")

document_cleaner = DocumentCleaner()
document_splitter = DocumentSplitter(split_by="word", split_length=150, split_overlap=50)
document_embedder = OpenAIDocumentEmbedder(api_key=Secret.from_token(openai_api_key))
document_writer = DocumentWriter(document_store)

pipe = Pipeline()
pipe.add_component("document_loader", document_loader)
pipe.add_component("document_cleaner", document_cleaner)
pipe.add_component("document_splitter", document_splitter)
pipe.add_component("document_embedder", document_embedder)
pipe.add_component("document_writer", document_writer)

pipe.connect("document_loader", "document_cleaner")
pipe.connect("document_cleaner", "document_splitter")
pipe.connect("document_splitter", "document_embedder")
pipe.connect("document_embedder", "document_writer")

print(
    "Running pipeline the Apify document_loader -> document_cleaner -> document_splitter "
    "-> document_embedder -> document_writer"
)
print("Crawling will take some time ...")
print("You can visit https://console.apify.com/actors/runs to monitor the progress")

pipe.run({"document_loader": {}})

print(f"Added {document_store.count_documents()} to vector from Website Content Crawler")

print("Retrieving documents from the document store using BM25")
print("query='Haystack'")
for doc in document_store.bm25_retrieval("Haystack", top_k=1):
    print(doc)

print("Retrieving documents from the document store using vector similarity")
print("query='What is Haystack'")
for doc in document_store.embedding_retrieval(OpenAITextEmbedder().run("Haystack")["embedding"], top_k=1):
    print(doc)
