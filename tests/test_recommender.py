import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.data_loader import SmartCityDataLoader
from src.recommender import SmartRecommender
from src.feature_engineering import HybridFeatureBuilder

class TestSmartRecommender(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.loader = SmartCityDataLoader('data/smart_city_items.csv', 'data/user_contexts.json')
        cls.recommender = SmartRecommender(cls.loader)
        cls.recommender.initialize()
        cls.recommender.compute_similarity_matrix(context='resident')
    
    def test_data_loading(self):
        """Test data loading functionality"""
        df = self.loader.load_items()
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)
        self.assertIn('item_id', df.columns)
        self.assertIn('name', df.columns)
        self.assertIn('category', df.columns)
    
    def test_preprocessing(self):
        """Test tag preprocessing"""
        self.loader.preprocess_tags()
        df = self.loader.items_df
        self.assertIn('tags_processed', df.columns)
        
        # Check that tags are properly processed
        sample_tags = df.iloc[0]['tags_processed']
        self.assertIsInstance(sample_tags, list)
        self.assertTrue(all(isinstance(tag, str) for tag in sample_tags))
    
    def test_feature_dimensions(self):
        """Test feature matrix dimensions"""
        feature_matrix = self.loader.create_feature_matrix()
        n_items = len(self.loader.items_df)
        self.assertEqual(feature_matrix.shape[0], n_items)
        self.assertGreater(feature_matrix.shape[1], 0)
    
    def test_cosine_similarity_range(self):
        """Test cosine similarity values are in [0,1] range"""
        similarity = self.recommender.similarity_matrix
        self.assertTrue(np.all(similarity >= 0))
        self.assertTrue(np.all(similarity <= 1))
    
    def test_self_similarity(self):
        """Test that self-similarity equals 1.0"""
        similarity = self.recommender.similarity_matrix
        for i in range(len(similarity)):
            self.assertAlmostEqual(similarity[i, i], 1.0, places=5)
    
    def test_valid_recommendations(self):
        """Test that recommendations are valid and exclude self"""
        test_item = self.recommender.item_names[0]
        recommendations = self.recommender.get_recommendations(test_item, top_n=3)
        
        self.assertIsNotNone(recommendations)
        self.assertGreater(len(recommendations), 0)
        
        # Check that recommendations don't include the input item
        rec_names = [rec['name'] for rec in recommendations]
        self.assertNotIn(test_item, rec_names)
        
        # Check that each recommendation has required fields
        for rec in recommendations:
            self.assertIn('name', rec)
            self.assertIn('similarity_score', rec)
            self.assertIn('category', rec)
    
    def test_context_switching(self):
        """Test that context switching changes results"""
        test_item = self.recommender.item_names[0]
        
        # Get recommendations for different contexts
        resident_recs = self.recommender.get_recommendations(test_item, context='resident', top_n=3)
        self.recommender.compute_similarity_matrix(context='emergency')
        emergency_recs = self.recommender.get_recommendations(test_item, context='emergency', top_n=3)
        
        # Check that they're different (not necessarily all different, but at least one)
        resident_names = [rec['name'] for rec in resident_recs]
        emergency_names = [rec['name'] for rec in emergency_recs]
        
        # Contexts should produce different recommendation sets
        self.assertNotEqual(resident_names, emergency_names)
    
    def test_invalid_item_name(self):
        """Test error handling for invalid item names"""
        with self.assertRaises(ValueError):
            self.recommender.get_recommendations("NonExistentItem")
    
    def test_category_recommendations(self):
        """Test category-based recommendations"""
        category = "Mobility"
        recommendations = self.recommender.get_category_based_recommendations(category, top_n=3)
        
        self.assertIsNotNone(recommendations)
        self.assertLessEqual(len(recommendations), 3)
        
        # Check all recommendations are from the requested category
        for rec in recommendations:
            self.assertEqual(rec['category'], category)
    
    def test_explanation_feature(self):
        """Test recommendation explanation"""
        item1 = self.recommender.item_names[0]
        item2 = self.recommender.item_names[1]
        
        explanation = self.recommender.explain_recommendation(item1, item2)
        
        self.assertIn('similarity_score', explanation)
        self.assertIn('explanation', explanation)
        self.assertIn('common_tags', explanation)
        self.assertIsInstance(explanation['similarity_score'], float)
    
    def test_hybrid_recommendations(self):
        """Test hybrid recommendation functionality"""
        test_item = self.recommender.item_names[0]
        hybrid_recs = self.recommender.get_hybrid_recommendations(test_item, popularity_weight=0.3, top_n=3)
        
        self.assertIsNotNone(hybrid_recs)
        self.assertGreater(len(hybrid_recs), 0)
        
        # Check hybrid scores are present
        for rec in hybrid_recs:
            self.assertIn('hybrid_score', rec)
            self.assertIsInstance(rec['hybrid_score'], float)

class TestFeatureEngineering(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.loader = SmartCityDataLoader('data/smart_city_items.csv', 'data/user_contexts.json')
        self.loader.load_items()
        self.loader.preprocess_tags()
        self.feature_builder = HybridFeatureBuilder()
    
    def test_tag_vectorization(self):
        """Test TF-IDF vectorization"""
        features = self.feature_builder.build_features(self.loader.items_df)
        self.assertIsNotNone(features)
        self.assertEqual(features.shape[0], len(self.loader.items_df))
    
    def test_feature_normalization(self):
        """Test feature vector normalization"""
        from src.feature_engineering import normalize_feature_vectors
        from scipy.sparse import random
        
        test_matrix = random(10, 20, density=0.5)
        normalized = normalize_feature_vectors(test_matrix)
        
        # Check L2 norm for each row
        import numpy as np
        norms = np.sqrt(np.sum(normalized.toarray()**2, axis=1))
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-5))

if __name__ == '__main__':
    unittest.main()