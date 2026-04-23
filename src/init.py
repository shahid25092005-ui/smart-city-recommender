"""
Smart City Recommendation Engine - Core Package
"""

from .data_loader import SmartCityDataLoader
from .feature_engineering import WeightedTagVectorizer, HybridFeatureBuilder
from .recommender import SmartRecommender
from .utils import evaluate_recommendations, visualize_similarity_heatmap, generate_report

__version__ = "1.0.0"
__all__ = [
    'SmartCityDataLoader',
    'WeightedTagVectorizer',
    'HybridFeatureBuilder',
    'SmartRecommender',
    'evaluate_recommendations',
    'visualize_similarity_heatmap',
    'generate_report'
]