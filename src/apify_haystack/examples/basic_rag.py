import urllib.request

from dotenv import load_dotenv
from haystack import Pipeline, PredefinedPipeline

load_dotenv()


urllib.request.urlretrieve("https://www.gutenberg.org/cache/epub/7785/pg7785.txt", "davinci.txt")

indexing_pipeline = Pipeline.from_template(PredefinedPipeline.INDEXING)
indexing_pipeline.run(data={"sources": ["davinci.txt"]})

rag_pipeline = Pipeline.from_template(PredefinedPipeline.RAG)

query = "How old was he when he died?"
result = rag_pipeline.run(data={"prompt_builder": {"query": query}, "text_embedder": {"text": query}})
print(result["llm"]["replies"][0])
