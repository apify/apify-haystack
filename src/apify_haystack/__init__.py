from importlib import metadata

__version__ = metadata.version("apify-haystack")

from apify_haystack.apify_dataset import ApifyDatasetFromActorCall, ApifyDatasetFromTaskCall, ApifyDatasetLoader

__all__ = ["ApifyDatasetLoader", "ApifyDatasetFromActorCall", "ApifyDatasetFromTaskCall"]
