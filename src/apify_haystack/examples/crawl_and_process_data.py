"""
Crawl websites, scrape text content, and store it in the InMemoryDocumentStore.

This script demonstrates how to extract content from a website using Apify's Website Content Crawler.
The content is split into smaller chunks, embedded, and stored in the InMemoryDocumentStore.

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

from haystack import Document, Pipeline
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.retrievers import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

from apify_haystack import ApifyDatasetFromActorCall

os.environ["APIFY_API_TOKEN"] = "YOUR-APIFY-API-TOKEN"
os.environ["OPENAI_API_KEY"] = "YOUR-OPENAI-API-KEY"

document_loader = ApifyDatasetFromActorCall(
    actor_id="apify/website-content-crawler",
    run_input={
        "maxCrawlPages": 3,  # limit the number of pages to crawl
        "startUrls": [{"url": "https://haystack.deepset.ai/"}],
    },
    dataset_mapping_function=lambda item: Document(content=item["text"] or "", meta={"url": item["url"]}),
)

document_store = InMemoryDocumentStore()
print(f"Initialized InMemoryDocumentStore with {document_store.count_documents()} documents")

document_splitter = DocumentSplitter(split_by="word", split_length=150, split_overlap=50)
document_embedder = OpenAIDocumentEmbedder()
document_writer = DocumentWriter(document_store)

pipe = Pipeline()
pipe.add_component("document_loader", document_loader)
pipe.add_component("document_splitter", document_splitter)
pipe.add_component("document_embedder", document_embedder)
pipe.add_component("document_writer", document_writer)

pipe.connect("document_loader", "document_splitter")
pipe.connect("document_splitter", "document_embedder")
pipe.connect("document_embedder", "document_writer")

print("\nCrawling will take some time ...")
print("You can visit https://console.apify.com/actors/runs to monitor the progress\n")

pipe.run({})
print(f"Added {document_store.count_documents()} to vector from Website Content Crawler")

print("\n ### Retrieving documents from the document store using BM25 ###\n")
print("query='Haystack'\n")

bm25_retriever = InMemoryBM25Retriever(document_store)

for doc in bm25_retriever.run("Haystack", top_k=1)["documents"]:
    print(doc.content)

print("\n ### Retrieving documents from the document store using vector similarity ###\n")
retrieval_pipe = Pipeline()
retrieval_pipe.add_component("embedder", OpenAITextEmbedder())
retrieval_pipe.add_component("retriever", InMemoryEmbeddingRetriever(document_store, top_k=1))

retrieval_pipe.connect("embedder.embedding", "retriever.query_embedding")

results = retrieval_pipe.run({"embedder": {"text": "What is Haystack?"}})

for doc in results["retriever"]["documents"]:
    print(doc.content)
