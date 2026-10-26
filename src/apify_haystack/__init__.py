from importlib import metadata

__version__ = metadata.version('apify-haystack')

from apify_haystack.apify_dataset import ApifyDatasetFromActorCall, ApifyDatasetFromTaskCall, ApifyDatasetLoader
from apify_haystack.web_search import ApifyWebSearch

__all__ = ['ApifyDatasetFromActorCall', 'ApifyDatasetFromTaskCall', 'ApifyDatasetLoader', 'ApifyWebSearch']
