"""
Retrieval-Augmented Generation (RAG): Extracting text content from a website and using it for question answering.

Steps involved:
- Load environment variables, including API keys.
- Define the Apify actor and its input parameters for web scraping.
- Initialize the Apify dataset loader to fetch and process the dataset.
- Set up the components for the RAG pipeline:
- Load documents from Apify and store their embeddings in the document store.
- Define a prompt template for the question-answering task.
- Build and connect the components in the pipeline.
- Run the pipeline to answer a sample question and print the response.

Expected output:
question: "What is haystack?"
......
Haystack is an open-source framework for building production-ready LLM applications
......
"""

import os

from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from apify_haystack import ApifyDatasetFromActorCall

# Set API keys here
os.environ["APIFY_API_TOKEN"] = "YOUR-APIFY-API-TOKEN"
os.environ["OPENAI_API_KEY"] = "YOUR-OPENAI-API-KEY"

actor_id = "apify/website-content-crawler"
run_input = {
    "maxCrawlPages": 1,  # limit the number of pages to crawl
    "startUrls": [{"url": "https://haystack.deepset.ai/"}],
}


def dataset_mapping_function(dataset_item: dict) -> Document:
    return Document(content=dataset_item.get("text"), meta={"url": dataset_item.get("url")})


apify_dataset_loader = ApifyDatasetFromActorCall(
    actor_id=actor_id,
    run_input=run_input,
    dataset_mapping_function=dataset_mapping_function,
)

# Components
print("Initializing components...")
document_store = InMemoryDocumentStore()

docs_embedder = OpenAIDocumentEmbedder()
text_embedder = OpenAITextEmbedder()
retriever = InMemoryEmbeddingRetriever(document_store)
generator = OpenAIGenerator(model="gpt-3.5-turbo")

# Load documents from Apify
print("Crawling will take some time ...")
print("You can visit https://console.apify.com/actors/runs to monitor the progress")
docs = apify_dataset_loader.run()
embeddings = docs_embedder.run(docs.get("documents"))
document_store.write_documents(embeddings["documents"])

template = """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""

prompt_builder = PromptBuilder(template=template)

# Add components to your pipeline
print("Initializing pipeline...")
pipe = Pipeline()
pipe.add_component("embedder", text_embedder)
pipe.add_component("retriever", retriever)
pipe.add_component("prompt_builder", prompt_builder)
pipe.add_component("llm", generator)

# Now, connect the components to each other
pipe.connect("embedder.embedding", "retriever.query_embedding")
pipe.connect("retriever", "prompt_builder.documents")
pipe.connect("prompt_builder", "llm")

question = "What is haystack?"


response = pipe.run({"embedder": {"text": question}, "prompt_builder": {"question": question}})

print(f"question: {question}")
print(f"answer: {response['llm']['replies'][0]}")

# Other questions
examples = [
    "Who created Haystack AI?",
    "Are there any upcoming events or community talks?",
]

for example in examples:
    response = pipe.run({"embedder": {"text": example}, "prompt_builder": {"question": example}})
    print(f"question: {question}")
    print(f"answer: {response['llm']['replies'][0]}")
