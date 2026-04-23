import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import json
import pickle
from pathlib import Path
from .feature_engineering import HybridFeatureBuilder, SimilarityCache
from .data_loader import SmartCityDataLoader
import warnings
warnings.filterwarnings('ignore')

class SmartRecommender:
    """Main recommendation engine for Smart City items"""
    
    def __init__(self, data_loader=None, feature_builder=None):
        self.data_loader = data_loader or SmartCityDataLoader()
        self.feature_builder = feature_builder or HybridFeatureBuilder()
        self.similarity_matrix = None
        self.cache = SimilarityCache()
        self.item_names = None
        
    def initialize(self):
        """Load data and build feature matrix"""
        self.data_loader.load_items()
        self.data_loader.preprocess_tags()
        self.data_loader.create_feature_matrix()
        self.feature_builder.build_features(self.data_loader.items_df)
        self.item_names = self.data_loader.get_all_item_names()
        
    def compute_similarity_matrix(self, method='cosine', context='resident', use_cache=True):
        """Compute similarity matrix using specified method and context"""
        
        if use_cache:
            cached_matrix = self.cache.get(method, context)
            if cached_matrix is not None:
                self.similarity_matrix = cached_matrix
                return self.similarity_matrix
        
        # Get context-weighted features
        weighted_features = self.data_loader.add_context_weights(context)
        
        # Compute similarity
        if method == 'cosine':
            similarity = cosine_similarity(weighted_features)
        else:
            raise ValueError(f"Method {method} not supported. Use 'cosine'")
        
        # Ensure similarity is in [0, 1] range
        similarity = np.clip(similarity, 0, 1)
        
        self.similarity_matrix = similarity
        
        if use_cache:
            self.cache.set(method, context, similarity)
        
        return similarity
    
    def get_recommendations(self, item_name, top_n=5, context='resident', exclude_self=True):
        """Get top N recommendations for an item with context awareness"""
        
        if self.similarity_matrix is None:
            self.compute_similarity_matrix(context=context)
        
        # Find item index
        try:
            item_idx = self.item_names.index(item_name)
        except ValueError:
            raise ValueError(f"Item '{item_name}' not found. Available items: {self.item_names[:10]}...")
        
        # Get similarity scores for the item
        scores = self.similarity_matrix[item_idx]
        
        # Create list of (index, score) pairs
        recommendations = list(enumerate(scores))
        
        # Exclude self if requested
        if exclude_self:
            recommendations = [(idx, score) for idx, score in recommendations if idx != item_idx]
        
        # Sort by similarity score (descending)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N
        top_recommendations = recommendations[:top_n]
        
        # Prepare results with metadata
        results = []
        for idx, score in top_recommendations:
            item = self.data_loader.items_df.iloc[idx]
            results.append({
                'name': item['name'],
                'category': item['category'],
                'similarity_score': round(score * 100, 2),  # Convert to percentage
                'location_zone': item['location_zone'],
                'is_24x7': item['is_24x7'],
                'tags': item['tags']
            })
        
        return results
    
    def explain_recommendation(self, item1, item2):
        """Explain why item2 is recommended for item1"""
        
        # Get both items
        item1_data = self.data_loader.get_item_by_name(item1)
        item2_data = self.data_loader.get_item_by_name(item2)
        
        if not item1_data or not item2_data:
            return {"error": "One or both items not found"}
        
        # Find common tags
        tags1 = set(item1_data['tags_processed']) if 'tags_processed' in item1_data else set(item1_data['tags'])
        tags2 = set(item2_data['tags_processed']) if 'tags_processed' in item2_data else set(item2_data['tags'])
        
        common_tags = tags1.intersection(tags2)
        
        # Check category match
        same_category = item1_data['category'] == item2_data['category']
        
        # Check location proximity
        same_zone = item1_data['location_zone'] == item2_data['location_zone']
        
        # Get similarity score
        idx1 = self.item_names.index(item1)
        idx2 = self.item_names.index(item2)
        similarity = self.similarity_matrix[idx1, idx2] * 100
        
        # Build explanation
        explanation_parts = []
        
        if common_tags:
            explanation_parts.append(f"Share {len(common_tags)} common features: {', '.join(list(common_tags)[:3])}")
        
        if same_category:
            explanation_parts.append(f"Same category: {item1_data['category']}")
        
        if same_zone:
            explanation_parts.append(f"Located in same zone: {item1_data['location_zone']}")
        
        if not explanation_parts:
            explanation_parts.append("Based on overall feature similarity")
        
        return {
            'similarity_score': round(similarity, 2),
            'explanation': " | ".join(explanation_parts),
            'common_tags': list(common_tags),
            'same_category': same_category,
            'same_zone': same_zone
        }
    
    def get_category_based_recommendations(self, category, top_n=3):
        """Get top N items from a specific category"""
        
        category_items = self.data_loader.items_df[self.data_loader.items_df['category'] == category]
        
        if len(category_items) == 0:
            return []
        
        # Sort by popularity score
        top_items = category_items.nlargest(top_n, 'popularity_score')
        
        return top_items[['name', 'category', 'popularity_score', 'location_zone']].to_dict('records')
    
    def get_hybrid_recommendations(self, item_name, popularity_weight=0.2, top_n=5, context='resident'):
        """Combine content similarity with popularity for recommendations"""
        
        # Get content-based recommendations
        content_recs = self.get_recommendations(item_name, top_n=top_n*2, context=context)
        
        # Apply popularity boost
        for rec in content_recs:
            item_data = self.data_loader.get_item_by_name(rec['name'])
            popularity = item_data['popularity_score']
            rec['hybrid_score'] = (rec['similarity_score'] / 100) * (1 - popularity_weight) + popularity * popularity_weight
        
        # Re-sort by hybrid score
        content_recs.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return content_recs[:top_n]
    
    def save_model(self, output_path='models/recommender_model.pkl'):
        """Save the trained model to disk"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'similarity_matrix': self.similarity_matrix,
            'item_names': self.item_names,
            'feature_builder': self.feature_builder,
            'data_loader': self.data_loader
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        return output_path
    
    def load_model(self, output_path='models/recommender_model.pkl'):
        """Load a trained model from disk"""
        with open(output_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.similarity_matrix = model_data['similarity_matrix']
        self.item_names = model_data['item_names']
        self.feature_builder = model_data['feature_builder']
        self.data_loader = model_data['data_loader']
        
        return self