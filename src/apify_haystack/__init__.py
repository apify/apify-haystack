from importlib import metadata

__version__ = metadata.version('apify-haystack')

from apify_haystack._actor_tool import ApifyActorTool
from apify_haystack._dataset import (
    ApifyDatasetFromActorCall,
    ApifyDatasetFromTaskCall,
    ApifyDatasetLoader,
    default_dataset_mapping_function,
)
from apify_haystack._web_search import ApifyWebSearch

__all__ = [
    'ApifyActorTool',
    'ApifyDatasetFromActorCall',
    'ApifyDatasetFromTaskCall',
    'ApifyDatasetLoader',
    'ApifyWebSearch',
    'default_dataset_mapping_function',
]
