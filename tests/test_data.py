import unittest
import pandas as pd
import json
from pathlib import Path

class TestDataIntegrity(unittest.TestCase):
    
    def setUp(self):
        """Load data files for testing"""
        self.items_df = pd.read_csv('data/smart_city_items.csv')
        with open('data/user_contexts.json', 'r') as f:
            self.contexts = json.load(f)
    
    def test_csv_structure(self):
        """Test that CSV has all required columns"""
        required_columns = [
            'item_id', 'name', 'category', 'tags', 'location_zone',
            'popularity_score', 'is_24x7', 'accessibility_features'
        ]
        
        for col in required_columns:
            self.assertIn(col, self.items_df.columns)
    
    def test_data_types(self):
        """Test correct data types for each column"""
        self.assertTrue(pd.api.types.is_integer_dtype(self.items_df['item_id']))
        self.assertTrue(pd.api.types.is_string_dtype(self.items_df['name']))
        self.assertTrue(pd.api.types.is_string_dtype(self.items_df['category']))
        self.assertTrue(pd.api.types.is_string_dtype(self.items_df['tags']))
        self.assertTrue(pd.api.types.is_string_dtype(self.items_df['location_zone']))
        self.assertTrue(pd.api.types.is_float_dtype(self.items_df['popularity_score']))
        self.assertTrue(pd.api.types.is_bool_dtype(self.items_df['is_24x7']))
    
    def test_popularity_scores(self):
        """Test popularity scores are within [0,1] range"""
        scores = self.items_df['popularity_score']
        self.assertTrue((scores >= 0).all())
        self.assertTrue((scores <= 1).all())
    
    def test_categories(self):
        """Test valid categories exist"""
        valid_categories = ['Mobility', 'Environment', 'Safety', 'Utility', 'Civic']
        categories = self.items_df['category'].unique()
        
        for category in categories:
            self.assertIn(category, valid_categories)
    
    def test_location_zones(self):
        """Test valid location zones"""
        valid_zones = ['north', 'south', 'east', 'west', 'central']
        zones = self.items_df['location_zone'].unique()
        
        for zone in zones:
            self.assertIn(zone, valid_zones)
    
    def test_unique_ids(self):
        """Test item_id uniqueness"""
        self.assertEqual(len(self.items_df['item_id'].unique()), len(self.items_df))
    
    def test_tags_format(self):
        """Test tags are properly formatted"""
        for tags in self.items_df['tags']:
            self.assertIsInstance(tags, str)
            tag_list = tags.split(',')
            self.assertGreater(len(tag_list), 0)
            # Each tag should be non-empty after stripping
            for tag in tag_list:
                self.assertGreater(len(tag.strip()), 0)
    
    def test_contexts_structure(self):
        """Test contexts JSON structure"""
        self.assertIn('context_weights', self.contexts)
        self.assertIn('context_descriptions', self.contexts)
        
        required_contexts = ['tourist', 'resident', 'emergency', 'night_mode']
        for context in required_contexts:
            self.assertIn(context, self.contexts['context_weights'])
            self.assertIn(context, self.contexts['context_descriptions'])
    
    def test_context_weights(self):
        """Test context weight values are valid"""
        for context, weights in self.contexts['context_weights'].items():
            self.assertIn('popularity_weight', weights)
            self.assertIn('accessibility_weight', weights)
            self.assertIn('category_preferences', weights)
            self.assertIn('tags_boost', weights)
            
            # Check weight ranges
            self.assertGreaterEqual(weights['popularity_weight'], 0)
            self.assertLessEqual(weights['popularity_weight'], 1)
            self.assertGreaterEqual(weights['accessibility_weight'], 0)
            self.assertLessEqual(weights['accessibility_weight'], 1)
    
    def test_data_completeness(self):
        """Test no missing critical values"""
        # Check for null values in critical columns
        critical_columns = ['item_id', 'name', 'category', 'location_zone']
        for col in critical_columns:
            self.assertFalse(self.items_df[col].isnull().any())
    
    def test_category_distribution(self):
        """Test each category has at least one item"""
        category_counts = self.items_df['category'].value_counts()
        for category in ['Mobility', 'Environment', 'Safety', 'Utility', 'Civic']:
            self.assertGreater(category_counts.get(category, 0), 0)
    
    def test_accessibility_features(self):
        """Test accessibility features format"""
        for features in self.items_df['accessibility_features']:
            if pd.notna(features):
                feature_list = str(features).split(',')
                for feature in feature_list:
                    feature = feature.strip()
                    self.assertIn(feature, ['wheelchair', 'braille', 'audio-announcement', 'none'])

if __name__ == '__main__':
    unittest.main()