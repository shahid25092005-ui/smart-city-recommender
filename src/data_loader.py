import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class SmartCityDataLoader:
    """Data loader for Smart City items with preprocessing and feature extraction"""
    
    def __init__(self, data_path='data/smart_city_items.csv', context_path='data/user_contexts.json'):
        self.data_path = data_path
        self.context_path = context_path
        self.items_df = None
        self.context_weights = None
        self.feature_matrix = None
        self.tfidf_vectorizer = None
        
    def load_items(self):
        """Load items from CSV file with proper parsing"""
        self.items_df = pd.read_csv(self.data_path)
        
        # Parse tags and accessibility features from strings to lists
        self.items_df['tags'] = self.items_df['tags'].apply(lambda x: [tag.strip() for tag in str(x).split(',')])
        self.items_df['accessibility_features'] = self.items_df['accessibility_features'].apply(
            lambda x: [feat.strip() for feat in str(x).split(',')] if pd.notna(x) else []
        )
        
        # Convert boolean column
        self.items_df['is_24x7'] = self.items_df['is_24x7'].astype(bool)
        
        return self.items_df
    
    def preprocess_tags(self):
        """Preprocess tags: lowercase, strip, remove duplicates"""
        self.items_df['tags_processed'] = self.items_df['tags'].apply(
            lambda tags: list(set([tag.lower().strip() for tag in tags]))
        )
        return self.items_df
    
    def create_feature_matrix(self, use_tfidf=True, max_features=100):
        """Create feature matrix using TF-IDF on tags and one-hot encoding on categories"""
        
        if self.items_df is None:
            self.load_items()
        
        self.preprocess_tags()
        
        # Create tag text for TF-IDF
        tag_texts = self.items_df['tags_processed'].apply(lambda x: ' '.join(x))
        
        if use_tfidf:
            self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
            tag_features = self.tfidf_vectorizer.fit_transform(tag_texts)
        else:
            # Simple count vectorizer fallback
            from sklearn.feature_extraction.text import CountVectorizer
            vectorizer = CountVectorizer(max_features=max_features)
            tag_features = vectorizer.fit_transform(tag_texts)
        
        # One-hot encode categories
        category_encoder = LabelEncoder()
        category_encoded = category_encoder.fit_transform(self.items_df['category'])
        category_features = pd.get_dummies(self.items_df['category'], prefix='cat')
        
        # One-hot encode location zones
        location_features = pd.get_dummies(self.items_df['location_zone'], prefix='zone')
        
        # Normalize popularity scores
        popularity_normalized = self.items_df['popularity_score'].values.reshape(-1, 1)
        
        # Combine all features
        from scipy.sparse import hstack
        import numpy as np
        
        # Convert categorical features to sparse
        from scipy.sparse import csr_matrix
        category_sparse = csr_matrix(category_features.values)
        
        location_sparse = csr_matrix(location_features.values)
        
        # Combine all features
        self.feature_matrix = hstack([
            tag_features,
            category_sparse,
            location_sparse,
            popularity_normalized
        ])
        
        return self.feature_matrix
    
    def add_context_weights(self, context_type='resident'):
        """Add context-specific weights to features"""
        if self.context_weights is None:
            with open(self.context_path, 'r') as f:
                context_data = json.load(f)
                self.context_weights = context_data['context_weights']
        
        if context_type not in self.context_weights:
            raise ValueError(f"Context type {context_type} not found. Available: {list(self.context_weights.keys())}")
        
        weights = self.context_weights[context_type]
        
        # Create context-aware feature weights
        if self.feature_matrix is None:
            self.create_feature_matrix()
            
        feature_weights = np.ones(self.feature_matrix.shape[1])
        
        # Apply popularity weight
        popularity_start_idx = self.feature_matrix.shape[1] - 1
        feature_weights[popularity_start_idx] = weights['popularity_weight']
        
        # Boost tags based on context
        tag_boost_tags = weights.get('tags_boost', [])
        if tag_boost_tags and hasattr(self, 'tfidf_vectorizer'):
            # Get indices of boosted tags in TF-IDF vocabulary
            boost_indices = []
            for tag in tag_boost_tags:
                if tag in self.tfidf_vectorizer.vocabulary_:
                    boost_indices.append(self.tfidf_vectorizer.vocabulary_[tag])
            
            for idx in boost_indices:
                if idx < self.feature_matrix.shape[1]:
                    feature_weights[idx] *= 1.5  # Boost tag importance
        
        # Apply weighted feature matrix
        weighted_matrix = self.feature_matrix.multiply(feature_weights)
        
        return weighted_matrix
    
    def get_item_metadata(self):
        """Return item metadata dictionary"""
        if self.items_df is None:
            self.load_items()
        
        metadata = self.items_df.to_dict('records')
        return metadata
    
    def get_item_by_name(self, name):
        """Get item details by name"""
        item = self.items_df[self.items_df['name'] == name]
        if len(item) == 0:
            return None
        return item.iloc[0].to_dict()
    
    def get_all_item_names(self):
        """Return list of all item names"""
        return self.items_df['name'].tolist()