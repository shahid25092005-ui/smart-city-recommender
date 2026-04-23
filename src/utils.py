import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import List, Dict, Any

def evaluate_recommendations(test_cases, recommender, metric='precision_at_k', k=3):
    """Evaluate recommendation quality using various metrics"""
    
    results = {}
    
    if metric == 'precision_at_k':
        # For precision, we need ground truth. Here we'll use category-based relevance
        precisions = []
        
        for test_item in test_cases:
            recs = recommender.get_recommendations(test_item, top_n=k)
            
            # Assume items in same category are relevant
            test_item_data = recommender.data_loader.get_item_by_name(test_item)
            if test_item_data:
                relevant_count = sum(1 for rec in recs if rec['category'] == test_item_data['category'])
                precision = relevant_count / k
                precisions.append(precision)
        
        results['precision_at_{}'.format(k)] = np.mean(precisions) if precisions else 0
    
    elif metric == 'recall_at_k':
        recalls = []
        
        for test_item in test_cases:
            recs = recommender.get_recommendations(test_item, top_n=k)
            
            test_item_data = recommender.data_loader.get_item_by_name(test_item)
            if test_item_data:
                # Total relevant items in same category (excluding self)
                same_category_items = recommender.data_loader.items_df[
                    (recommender.data_loader.items_df['category'] == test_item_data['category']) &
                    (recommender.data_loader.items_df['name'] != test_item)
                ]
                total_relevant = len(same_category_items)
                
                if total_relevant > 0:
                    relevant_retrieved = sum(1 for rec in recs if rec['category'] == test_item_data['category'])
                    recall = relevant_retrieved / total_relevant
                    recalls.append(recall)
        
        results['recall_at_{}'.format(k)] = np.mean(recalls) if recalls else 0
    
    elif metric == 'ndcg':
        # Normalized Discounted Cumulative Gain
        ndcgs = []
        
        for test_item in test_cases:
            recs = recommender.get_recommendations(test_item, top_n=k)
            
            test_item_data = recommender.data_loader.get_item_by_name(test_item)
            if test_item_data:
                # Relevance scores: 2 for same category, 1 for same zone, 0 otherwise
                relevance = []
                for rec in recs:
                    score = 0
                    if rec['category'] == test_item_data['category']:
                        score = 2
                    elif rec['location_zone'] == test_item_data['location_zone']:
                        score = 1
                    relevance.append(score)
                
                # Calculate DCG
                dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance))
                
                # Calculate IDCG (ideal ordering)
                ideal_relevance = sorted(relevance, reverse=True)
                idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevance))
                
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcgs.append(ndcg)
        
        results['ndcg@{}'.format(k)] = np.mean(ndcgs) if ndcgs else 0
    
    return results

def visualize_similarity_heatmap(similarity_matrix, item_names, save_path='outputs/similarity_heatmap.png'):
    """Create and save a heatmap visualization of the similarity matrix"""
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap
    im = ax.imshow(similarity_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Similarity Score')
    
    # Configure axes
    ax.set_xticks(np.arange(len(item_names)))
    ax.set_yticks(np.arange(len(item_names)))
    ax.set_xticklabels(item_names, rotation=90, fontsize=8)
    ax.set_yticklabels(item_names, fontsize=8)
    
    # Add labels
    ax.set_xlabel('Items', fontsize=12)
    ax.set_ylabel('Items', fontsize=12)
    ax.set_title('Item Similarity Matrix', fontsize=14, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path

def generate_report(recommendations_dict, output_file='outputs/recommendations_report.json'):
    """Generate a comprehensive report of recommendations"""
    
    report = {
        'summary': {
            'total_queries': len(recommendations_dict),
            'average_recommendations': np.mean([len(recs) for recs in recommendations_dict.values()]) if recommendations_dict else 0
        },
        'recommendations': recommendations_dict,
        'statistics': {}
    }
    
    # Calculate category distribution
    all_categories = []
    for recs in recommendations_dict.values():
        for rec in recs:
            if isinstance(rec, dict) and 'category' in rec:
                all_categories.append(rec['category'])
    
    if all_categories:
        from collections import Counter
        category_counts = Counter(all_categories)
        report['statistics']['category_distribution'] = dict(category_counts)
        report['statistics']['most_common_category'] = category_counts.most_common(1)[0][0] if category_counts else None
    
    # Save report
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    return output_file

def validate_input(item_name, available_items):
    """Validate user input item name"""
    
    if not isinstance(item_name, str):
        return False, "Item name must be a string"
    
    if not item_name.strip():
        return False, "Item name cannot be empty"
    
    if item_name not in available_items:
        return False, f"Item '{item_name}' not found. Please select from available items"
    
    return True, "Valid input"

def calculate_diversity(recommendations):
    """Calculate diversity of recommendations based on categories"""
    
    if not recommendations:
        return 0
    
    categories = [rec.get('category') for rec in recommendations if isinstance(rec, dict)]
    unique_categories = len(set(categories))
    
    diversity = unique_categories / len(categories) if categories else 0
    return diversity

def format_recommendations_for_display(recommendations):
    """Format recommendations for display in UI"""
    
    formatted = []
    for i, rec in enumerate(recommendations, 1):
        formatted.append({
            'rank': i,
            'name': rec.get('name', 'Unknown'),
            'similarity': rec.get('similarity_score', 0),
            'category': rec.get('category', 'N/A'),
            'location': rec.get('location_zone', 'N/A'),
            'is_24x7': rec.get('is_24x7', False)
        })
    
    return formatted