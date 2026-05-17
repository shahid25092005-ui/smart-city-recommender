import streamlit as st
import pandas as pd
import json
from pathlib import Path
import sys
import textwrap

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import SmartCityDataLoader
from src.recommender import SmartRecommender

# Page configuration
st.set_page_config(
    page_title="MallRec - Intelligent Mall Discovery",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium Custom CSS
st.markdown("""
<style>
    /* Base styling */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, p, span, label {
        color: #f8fafc !important;
    }

    /* Main Header with dynamic gradient */
    .hero-section {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        padding: 3rem;
        border-radius: 24px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(rgba(255,255,255,0.1), rgba(255,255,255,0));
        pointer-events: none;
    }

    .hero-title {
        font-size: 3rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        font-size: 1.2rem !important;
        opacity: 0.9;
        max-width: 600px;
    }

    /* Glassmorphism containers */
    .glass-container {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    /* Recommendation Cards */
    .rec-card {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .rec-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
        border-color: rgba(139, 92, 246, 0.5);
    }

    .rec-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 4px; height: 100%;
        background: linear-gradient(to bottom, #3b82f6, #8b5cf6);
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    .badge-primary { background: rgba(59, 130, 246, 0.2); color: #60a5fa !important; border: 1px solid rgba(59, 130, 246, 0.3); }
    .badge-secondary { background: rgba(139, 92, 246, 0.2); color: #a78bfa !important; border: 1px solid rgba(139, 92, 246, 0.3); }
    .badge-accent { background: rgba(236, 72, 153, 0.2); color: #f472b6 !important; border: 1px solid rgba(236, 72, 153, 0.3); }
    .badge-success { background: rgba(16, 185, 129, 0.2); color: #34d399 !important; border: 1px solid rgba(16, 185, 129, 0.3); }

    /* Match Pill */
    .match-pill {
        background: rgba(255, 255, 255, 0.1);
        padding: 0.25rem 0.75rem;
        border-radius: 8px;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        margin: 0.25rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .match-pill.matched {
        background: rgba(16, 185, 129, 0.15);
        border-color: rgba(16, 185, 129, 0.4);
        color: #34d399 !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        box-shadow: 0 10px 15px -3px rgba(139, 92, 246, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }

    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 15px 20px -3px rgba(139, 92, 246, 0.5) !important;
        opacity: 0.95 !important;
    }

    /* Score Circle */
    .score-circle {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1.2rem;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'recommender' not in st.session_state:
    @st.cache_resource
    def load_recommender():
        loader = SmartCityDataLoader('data/shopping_malls.csv', 'data/user_contexts.json')
        # We need to initialize loader data to get features
        loader.load_items()
        loader.preprocess_tags()
        recommender = SmartRecommender(loader)
        return recommender
    
    st.session_state.recommender = load_recommender()

if 'features' not in st.session_state:
    st.session_state.features = st.session_state.recommender.data_loader.get_unique_features()

# Header
st.markdown("""
<div class="hero-section">
    <div class="hero-title">✨ Intelligent Mall Discovery</div>
    <div class="hero-subtitle">Tell us what you're looking for. We'll find the perfect destination.</div>
</div>
""", unsafe_allow_html=True)

# Selection Interface
st.markdown('<div class="glass-container">', unsafe_allow_html=True)
st.markdown("### 🎯 Craft Your Perfect Outing")

col1, col2 = st.columns(2)

with col1:
    selected_tags = st.multiselect(
        "🏷️ Desired Features & Vibe",
        options=st.session_state.features['tags'],
        placeholder="e.g., premium-brands, food-court, easy-parking..."
    )
    
    selected_activities = st.multiselect(
        "🎮 Entertainment & Activities",
        options=st.session_state.features['activities'],
        placeholder="e.g., Cinepolis Cinemas, Bowling Alley..."
    )

with col2:
    selected_brands = st.multiselect(
        "🛍️ Specific Brands",
        options=st.session_state.features['brands'],
        placeholder="e.g., Zara, H&M, Reliance Digital..."
    )
    
    selected_restaurants = st.multiselect(
        "🍔 Cafes & Restaurants",
        options=st.session_state.features['restaurants'],
        placeholder="e.g., Starbucks, KFC, Barbeque Nation..."
    )

st.markdown('</div>', unsafe_allow_html=True)

# Action Button
if st.button("🚀 Discover Your Perfect Malls", use_container_width=True):
    total_selections = len(selected_tags) + len(selected_activities) + len(selected_brands) + len(selected_restaurants)
    
    if total_selections == 0:
        st.warning("Please select at least one feature, activity, brand, or restaurant to get personalized recommendations!")
    else:
        with st.spinner("Analyzing matches..."):
            recommendations = st.session_state.recommender.get_recommendations_by_features(
                selected_tags=selected_tags,
                selected_brands=selected_brands,
                selected_restaurants=selected_restaurants,
                selected_activities=selected_activities,
                top_n=5
            )
            
            st.markdown("<br><h2>🏆 Top Recommended Malls For You</h2>", unsafe_allow_html=True)
            
            if not recommendations or recommendations[0]['similarity_score'] == 0:
                st.info("No malls strongly matched your specific criteria. Try broadening your search!")
            
            for idx, rec in enumerate(recommendations):
                # Only show top 5 and skip 0 scores if they exist
                if rec['similarity_score'] == 0:
                    continue
                    
                # Format matched items as pills
                matches = rec['matched_features']
                matches_html = ""
                
                if matches['tags']:
                    matches_html += f"<div><small style='color:#94a3b8'>Tags:</small> " + " ".join([f"<span class='match-pill matched'>✓ {t}</span>" for t in matches['tags']]) + "</div>"
                if matches['brands']:
                    matches_html += f"<div><small style='color:#94a3b8'>Brands:</small> " + " ".join([f"<span class='match-pill matched'>✓ {b}</span>" for b in matches['brands']]) + "</div>"
                if matches['restaurants']:
                    matches_html += f"<div><small style='color:#94a3b8'>Dining:</small> " + " ".join([f"<span class='match-pill matched'>✓ {r}</span>" for r in matches['restaurants']]) + "</div>"
                if matches['activities']:
                    matches_html += f"<div><small style='color:#94a3b8'>Activities:</small> " + " ".join([f"<span class='match-pill matched'>✓ {a}</span>" for a in matches['activities']]) + "</div>"
                
                if not matches_html:
                    matches_html = "<div style='color:#94a3b8; font-style: italic;'>Recommended based on general popularity</div>"

                is_24x7_badge = '<span class="badge badge-success">24x7 Open</span>' if rec['is_24x7'] else ''
                card_html = f"""<div class="rec-card">
<div style="display: flex; justify-content: space-between; align-items: flex-start;">
<div>
<h2 style="margin:0; font-size: 1.8rem; color: white !important;">{idx+1}. {rec['name']}</h2>
<div style="margin-top: 0.5rem; margin-bottom: 1rem;">
<span class="badge badge-primary">{rec['category']}</span>
<span class="badge badge-secondary">📍 {rec['location_zone']} ({rec['distance_km']}km)</span>
{is_24x7_badge}
</div>
</div>
<div class="score-circle">
{int(rec['similarity_score'])}
</div>
</div>
<div style="background: rgba(15, 23, 42, 0.5); border-radius: 12px; padding: 1.2rem; margin-top: 1rem;">
<h4 style="margin-top: 0; margin-bottom: 0.8rem; color: #cbd5e1;">🎯 Your Matched Criteria:</h4>
{matches_html}
</div>
<div style="margin-top: 1rem; color: #94a3b8; font-size: 0.9rem;">
<strong>Highlights:</strong> {", ".join(rec['tags'][:5])}...
</div>
</div>"""
                
                st.markdown(card_html, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; color: #64748b; margin-top: 4rem; padding-top: 2rem; border-top: 1px solid rgba(255,255,255,0.1);">
    <p>MallRec v2.0 | Intelligent Feature-Based Discovery</p>
</div>
""", unsafe_allow_html=True)