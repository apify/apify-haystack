"""
Analyze Your Instagram Comments' Vibe with Apify and Haystack

This script demonstrates how to extract comments from an Instagram post using Apify's Instagram Scraper.
The content is processed using the Haystack pipeline and the vibe of the comments is analyzed using the LLM model.

URL = "https://www.instagram.com/p/C_a9jcRuJZZ/"  # @tiffintech on How to easily keep up with tech?

Expected output:
......
Overall, the Instagram post seems to be generating positive energy and high engagement. The comments are filled with
emojis like 😍, 😂, ❤️, 🙏🏿, and 🙌 which show excitement and enthusiasm .....
.... Overall, the engagement patterns suggest that the post is vibrating with high energy and positivity.
"""

import os

from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.preprocessors import DocumentCleaner

from apify_haystack import ApifyDatasetFromActorCall

os.environ["APIFY_API_TOKEN"] = "YOUR-APIFY-API-TOKEN"
os.environ["OPENAI_API_KEY"] = "YOUR-OPENAI-API-KEY"


def dataset_mapping_function(dataset_item: dict) -> Document:
    return Document(
        content=dataset_item.get("text"),
        meta={"ownerUsername": dataset_item.get("ownerUsername")},
    )


document_loader = ApifyDatasetFromActorCall(
    actor_id="apify/instagram-comment-scraper",
    run_input={"resultsLimit": 50},
    dataset_mapping_function=dataset_mapping_function,
)

prompt = """
Analyze these Instagram comments to determine if the post is generating positive energy, excitement,
or high engagement. Focus on sentiment, emotional tone, and engagement patterns to conclude if
the post is 'vibrating' with high energy. Be concise."

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Analysis:
"""

cleaner = DocumentCleaner(
    remove_empty_lines=True,
    remove_extra_whitespaces=True,
    remove_repeated_substrings=True,
)
prompt_builder = PromptBuilder(template=prompt)
generator = OpenAIGenerator(model="gpt-3.5-turbo")


pipe = Pipeline()
pipe.add_component("loader", document_loader)
pipe.add_component("cleaner", cleaner)
pipe.add_component("prompt_builder", PromptBuilder(template=prompt))
pipe.add_component("llm", OpenAIGenerator(model="gpt-3.5-turbo"))
pipe.connect("loader", "cleaner")
pipe.connect("cleaner", "prompt_builder")
pipe.connect("prompt_builder", "llm")

print("Vibe analysis:\n")

url1 = "https://www.instagram.com/p/C_a9jcRuJZZ/"  # @tiffintech on How to easily keep up with tech?
print(f"\nScraping Instagram comments for: {url1} and running analysis (it might take around 30-60 secs) ...\n")

res = pipe.run({"loader": {"run_input": {"directUrls": [url1]}}})
print("Analysis:", res.get("llm", {}).get("replies", "No response")[0])

url2 = "https://www.instagram.com/p/C_RgBzogufK/"  # @maharishis on Affordable Care Act
print(f"\nScraping Instagram comments for: {url2} and running analysis (it might take around 30-60 secs)... \n")

res = pipe.run({"loader": {"run_input": {"directUrls": [url2]}}})
print("Analysis:", res.get("llm", {}).get("replies", "No response")[0])
