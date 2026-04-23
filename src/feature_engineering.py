import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix, hstack
import hashlib
import pickle
from functools import lru_cache

class WeightedTagVectorizer:
    """TF-IDF based tag vectorizer with weighting support"""
    
    def __init__(self, max_features=100, use_idf=True):
        self.max_features = max_features
        self.use_idf = use_idf
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            use_idf=use_idf,
            stop_words='english',
            lowercase=True,
            token_pattern=r'(?u)\b\w+\b'
        )
        self.tag_weights = None
        
    def fit_transform(self, tag_lists):
        """Fit and transform tag lists to TF-IDF matrix"""
        tag_texts = [' '.join(tags) for tags in tag_lists]
        return self.vectorizer.fit_transform(tag_texts)
    
    def transform(self, tag_lists):
        """Transform tag lists using fitted vectorizer"""
        tag_texts = [' '.join(tags) for tags in tag_lists]
        return self.vectorizer.transform(tag_texts)
    
    def get_feature_names(self):
        """Get feature names (tags)"""
        return self.vectorizer.get_feature_names_out()
    
    def apply_weights(self, tfidf_matrix, context_weights):
        """Apply context-specific weights to TF-IDF features"""
        if context_weights is None:
            return tfidf_matrix
        
        weighted_matrix = tfidf_matrix.copy()
        for tag, weight in context_weights.items():
            if tag in self.vectorizer.vocabulary_:
                idx = self.vectorizer.vocabulary_[tag]
                weighted_matrix[:, idx] *= weight
        
        return weighted_matrix

class HybridFeatureBuilder:
    """Combine multiple feature types for comprehensive item representation"""
    
    def __init__(self, tag_weight=0.5, category_weight=0.4, location_weight=0.1):
        self.tag_weight = tag_weight
        self.category_weight = category_weight
        self.location_weight = location_weight
        self.tag_vectorizer = None
        self.category_encoder = None
        self.location_encoder = None
        self.normalized_features = None
        
    def build_features(self, items_df):
        """Build hybrid feature matrix combining tags, categories, and locations"""
        
        # 1. Tag features (TF-IDF)
        self.tag_vectorizer = WeightedTagVectorizer(max_features=100)
        tag_features = self.tag_vectorizer.fit_transform(items_df['tags_processed'])
        
        # 2. Category features (One-hot encoding)
        category_features = pd.get_dummies(items_df['category'], prefix='cat')
        category_matrix = csr_matrix(category_features.values)
        
        # 3. Location features (One-hot encoding)
        location_features = pd.get_dummies(items_df['location_zone'], prefix='zone')
        location_matrix = csr_matrix(location_features.values)
        
        # Apply weights to each feature type
        weighted_tag = tag_features * self.tag_weight
        weighted_category = category_matrix * self.category_weight
        weighted_location = location_matrix * self.location_weight
        
        # Combine all features
        hybrid_matrix = hstack([weighted_tag, weighted_category, weighted_location])
        
        # Normalize the combined matrix
        self.normalized_features = normalize(hybrid_matrix, norm='l2')
        
        return self.normalized_features
    
    def get_feature_importance(self):
        """Return feature importance weights"""
        return {
            'tag_weight': self.tag_weight,
            'category_weight': self.category_weight,
            'location_weight': self.location_weight
        }
    
    def update_weights(self, tag_weight=None, category_weight=None, location_weight=None):
        """Update feature weights dynamically"""
        if tag_weight is not None:
            self.tag_weight = tag_weight
        if category_weight is not None:
            self.category_weight = category_weight
        if location_weight is not None:
            self.location_weight = location_weight
        
        # Normalize weights to sum to 1
        total = self.tag_weight + self.category_weight + self.location_weight
        self.tag_weight /= total
        self.category_weight /= total
        self.location_weight /= total

def normalize_feature_vectors(feature_matrix, norm='l2'):
    """Normalize feature vectors using specified norm"""
    return normalize(feature_matrix, norm=norm)

class SimilarityCache:
    """Cache similarity computations for performance"""
    
    def __init__(self, cache_size=100):
        self.cache = {}
        self.cache_size = cache_size
        self.access_count = {}
        
    def get_cache_key(self, method, context):
        """Generate cache key from parameters"""
        key_string = f"{method}_{context}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, method, context):
        """Retrieve cached similarity matrix"""
        key = self.get_cache_key(method, context)
        return self.cache.get(key)
    
    def set(self, method, context, similarity_matrix):
        """Cache similarity matrix with LRU eviction"""
        key = self.get_cache_key(method, context)
        
        if len(self.cache) >= self.cache_size:
            # Remove least recently used
            lru_key = min(self.access_count, key=self.access_count.get)
            del self.cache[lru_key]
            del self.access_count[lru_key]
        
        self.cache[key] = similarity_matrix
        self.access_count[key] = 0
    
    def increment_access(self, method, context):
        """Increment access count for LRU tracking"""
        key = self.get_cache_key(method, context)
        if key in self.access_count:
            self.access_count[key] += 1