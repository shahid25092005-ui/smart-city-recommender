import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import SmartCityDataLoader
from src.recommender import SmartRecommender
from src.utils import format_recommendations_for_display

# Page configuration
st.set_page_config(
    page_title="SmartRec - Smart City Recommendation Engine",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp span, .stApp label {
        color: #1e293b !important;
    }
    .main-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white !important;
        margin-bottom: 2.5rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .main-header h1, .main-header h2, .main-header h3, .main-header p {
        color: white !important;
    }
    .recommendation-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        border: 1px solid #e2e8f0;
    }
    .recommendation-card h3 {
        color: #1e293b !important;
        margin-top: 0;
    }
    .recommendation-card p {
        color: #475569 !important;
    }
    .recommendation-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    /* Sidebar Visibility Fixes */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown p {
        color: #e2e8f0 !important;
        font-weight: 500;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    /* Info box in sidebar */
    [data-testid="stSidebar"] [data-testid="stNotification"] {
        background-color: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid #334155 !important;
    }
    [data-testid="stSidebar"] [data-testid="stNotification"] [data-testid="stMarkdownContainer"] p {
        color: #94a3b8 !important;
    }
    /* Expander in sidebar */
    [data-testid="stSidebar"] .st-ae {
        border-color: #334155 !important;
    }
    [data-testid="stSidebar"] .st-ae p {
        color: #cbd5e1 !important;
    }
    /* Global Button Styles */
    .stButton>button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        background-color: #ffffff !important;
        color: #1e293b !important;
        border: 1px solid #e2e8f0 !important;
        padding: 0.5rem 1rem !important;
        width: 100%;
    }
    .stButton>button:hover {
        border-color: #4f46e5 !important;
        color: #4f46e5 !important;
        background-color: #f8fafc !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }
    /* Primary buttons (Get Recommendations) */
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: white !important;
        border: none !important;
    }
    .stButton>button[kind="primary"]:hover {
        opacity: 0.9 !important;
        color: white !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important;
    }
    .similarity-score {
        font-size: 1.5rem;
        font-weight: bold;
        color: #4f46e5;
    }
    .category-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    .location-badge {
        background-color: #48bb78;
        color: white;
    }
    .safety-badge {
        background-color: #f56565;
        color: white;
    }
    .context-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #4f46e5;
        margin-bottom: 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    /* Nav bar buttons */
    .nav-btn-container {
        display: flex;
        gap: 0.5rem;
        background-color: white;
        padding: 0.6rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }
    .stButton > button.nav-btn-active {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: white !important;
        box-shadow: 0 8px 15px rgba(79, 70, 229, 0.4) !important;
        border: none !important;
    }
    .stButton > button.nav-btn-inactive {
        background-color: transparent !important;
        color: #64748b !important;
        border: 1px solid transparent !important;
    }
    .stButton > button.nav-btn-inactive:hover {
        color: #4f46e5 !important;
        background-color: #f8fafc !important;
        border-color: #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'recommender' not in st.session_state:
    @st.cache_resource
    def load_recommender():
        loader = SmartCityDataLoader('data/smart_city_items.csv', 'data/user_contexts.json')
        recommender = SmartRecommender(loader)
        recommender.initialize()
        return recommender
    
    st.session_state.recommender = load_recommender()
    st.session_state.current_context = 'resident'

# Sidebar
with st.sidebar:
    st.markdown('<div style="text-align: center; padding-bottom: 1rem;">', unsafe_allow_html=True)
    st.markdown("### 🏙️ SmartRec")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    # About section
    with st.expander("ℹ️ About SmartRec"):
        st.markdown("""
        **SmartRec** is a context-aware recommendation engine for Smart City infrastructure.
        
        **Features:**
        - Content-based filtering with TF-IDF
        - Context-aware recommendations
        - Hybrid scoring (similarity + popularity)
        - Real-time explanation generation
        
        **Categories:** Mobility, Environment, Safety, Utility, Civic
        """)

# Main content
st.markdown(f"""
<div class="main-header">
    <h1>🏙️ Smart City Recommendation Engine</h1>
    <p>Discover the perfect Smart City services tailored to your needs</p>
    <div style="display: flex; gap: 2rem; margin-top: 1.5rem;">
        <div>
            <small style="opacity: 0.8;">Total Infrastructure Items</small>
            <h2 style="margin: 0; color: white !important;">{len(st.session_state.recommender.data_loader.items_df)}</h2>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation Bar for Context
context_options = {
    'resident': "🏠 Resident",
    'tourist': "✈️ Tourist",
    'emergency': "🚨 Emergency",
    'night_mode': "🌙 Night Mode"
}

st.write("🎯 **Select Your Context:**")
nav_cols = st.columns(len(context_options))
for i, (key, label) in enumerate(context_options.items()):
    is_active = st.session_state.current_context == key
    if nav_cols[i].button(
        label, 
        key=f"nav_{key}", 
        use_container_width=True,
    ):
        if st.session_state.current_context != key:
            st.session_state.current_context = key
            st.session_state.recommender.compute_similarity_matrix(context=key)
            st.rerun()

# Apply active/inactive styles using a small hack since Streamlit buttons are hard to style individually
# We use the key to inject specific styles
for idx, key in enumerate(context_options.keys()):
    is_active = st.session_state.current_context == key
    bg_style = "linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important" if is_active else "white !important"
    text_color = "white !important" if is_active else "#64748b !important"
    shadow = "0 8px 15px rgba(79, 70, 229, 0.4) !important" if is_active else "none !important"
    border = "none !important" if is_active else "1px solid #e2e8f0 !important"
    
    st.markdown(f"""
        <style>
        div[data-testid="stHorizontalBlock"] div:nth-child({idx + 1}) button {{
            background: {bg_style};
            color: {text_color};
            box-shadow: {shadow};
            border: {border};
            transition: all 0.3s ease !important;
        }}
        div[data-testid="stHorizontalBlock"] div:nth-child({idx + 1}) button:hover {{
            transform: translateY(-2px);
            {"box-shadow: 0 10px 20px rgba(79, 70, 229, 0.5) !important;" if is_active else "border-color: #4f46e5 !important; color: #4f46e5 !important;"}
        }}
        </style>
    """, unsafe_allow_html=True)

# Context description card
context_descriptions = {
    'resident': "🏠 Resident: Focused on daily commuting, practical services, and long-term utility.",
    'tourist': "✈️ Tourist: Prioritizing city amenities, landmarks, and leisure activities.",
    'emergency': "🚨 Emergency: Immediate access to safety-critical infrastructure and medical services.",
    'night_mode': "🌙 Night Mode: Highlighting 24x7 services and safety-first night infrastructure."
}

if st.session_state.current_context:
    st.markdown(f"""
    <div class="context-card">
        <div style="font-size: 1.1rem; color: #1e293b;">
            {context_descriptions[st.session_state.current_context]}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Search and selection
st.subheader("🔍 Select a Smart City Item")
all_items = st.session_state.recommender.data_loader.get_all_item_names()
selected_item = st.selectbox("Choose an item to get recommendations:", all_items)

# Get item metadata
item_metadata = st.session_state.recommender.data_loader.get_item_by_name(selected_item)

if item_metadata:
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Category", item_metadata['category'])
    with col2:
        st.metric("Distance", f"{item_metadata['distance_km']} km")
    with col3:
        st.metric("Location", item_metadata['location_zone'].title())
    with col4:
        st.metric("Popularity", f"{item_metadata['popularity_score']*100:.0f}%")
    with col5:
        st.metric("24x7", "Yes" if item_metadata['is_24x7'] else "No")
    
    st.markdown(f"**Tags:** {', '.join(item_metadata['tags'])}")
    if item_metadata['accessibility_features']:
        st.markdown(f"**Accessibility:** {', '.join(item_metadata['accessibility_features'])}")

# Recommendation type selector
rec_type = st.radio(
    "Recommendation Type:",
    ['Content-Based', 'Hybrid (Popularity Boost)'],
    horizontal=True
)

# Get recommendations
if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
    with st.spinner("Generating recommendations..."):
        if rec_type == 'Content-Based':
            recommendations = st.session_state.recommender.get_recommendations(
                selected_item, 
                top_n=5, 
                context=st.session_state.current_context
            )
        else:
            recommendations = st.session_state.recommender.get_hybrid_recommendations(
                selected_item, 
                popularity_weight=0.3, 
                top_n=5,
                context=st.session_state.current_context
            )
        
        st.session_state.recommendations = recommendations

# Display recommendations
if 'recommendations' in st.session_state and st.session_state.recommendations:
    st.subheader("📊 Top Recommendations")
    
    cols = st.columns(2)
    for idx, rec in enumerate(st.session_state.recommendations):
        with cols[idx % 2]:
                is_24x7_badge = '<span class="category-badge safety-badge">24x7</span>' if rec['is_24x7'] else ''
                card_html = f'<div class="recommendation-card">' \
                            f'<h3>{idx + 1}. {rec["name"]}</h3>' \
                            f'<div>' \
                            f'<span class="category-badge" style="background-color: #667eea; color: white;">{rec["category"]}</span>' \
                            f'<span class="category-badge location-badge">{rec["location_zone"].upper()}</span>' \
                            f'{is_24x7_badge}' \
                            f'</div>' \
                            f'<div class="similarity-score">{rec["similarity_score"]:.1f}%</div>' \
                            f'<small style="color: #64748b;">Match Score | 📍 {rec["distance_km"]} km away</small>' \
                            f'<p style="margin-top: 10px;"><strong>Tags:</strong> {", ".join(rec["tags"][:3])}</p>' \
                            f'</div>'
                st.markdown(card_html, unsafe_allow_html=True)
    
    # Explanation button
    if st.button("💡 Explain Top Recommendation", use_container_width=True):
        top_rec = st.session_state.recommendations[0]
        explanation = st.session_state.recommender.explain_recommendation(selected_item, top_rec['name'])
        
        st.success(f"""
        **Why is {top_rec['name']} recommended?**
        
        Similarity Score: {explanation['similarity_score']:.1f}%
        
        {explanation['explanation']}
        
        {'✅ Same category' if explanation['same_category'] else '🔄 Different category but similar features'}
        {'📍 Same zone' if explanation['same_zone'] else '📍 Different zone'}
        """)
    
    # Download button
    if st.button("📥 Download Recommendations (JSON)", use_container_width=True):
        rec_data = {
            'input_item': selected_item,
            'context': st.session_state.current_context,
            'recommendations': st.session_state.recommendations
        }
        st.download_button(
            label="Download JSON",
            data=json.dumps(rec_data, indent=2),
            file_name=f"recommendations_{selected_item.replace(' ', '_')}.json",
            mime="application/json"
        )

# Visualization section
st.markdown("---")
st.subheader("📈 Similarity Visualization")

if st.button("Generate Similarity Heatmap"):
    with st.spinner("Generating visualization..."):
        import plotly.graph_objects as go
        import numpy as np
        
        # Get top 10 items for visualization
        top_items = st.session_state.recommender.data_loader.items_df.nlargest(10, 'popularity_score')['name'].tolist()
        top_indices = [st.session_state.recommender.item_names.index(item) for item in top_items]
        similarity_subset = st.session_state.recommender.similarity_matrix[np.ix_(top_indices, top_indices)]
        
        fig = go.Figure(data=go.Heatmap(
            z=similarity_subset,
            x=top_items,
            y=top_items,
            colorscale='Viridis',
            text=similarity_subset.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar_title="Similarity Score"
        ))
        
        fig.update_layout(
            title="Similarity Matrix - Top 10 Popular Items",
            xaxis_title="Items",
            yaxis_title="Items",
            width=700,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>SmartRec v1.0 | Context-Aware Recommendation Engine for Smart Cities</p>
        <p>🏙️ Mobility | 🌿 Environment | 🚨 Safety | 💡 Utility | 📚 Civic</p>
    </div>
    """,
    unsafe_allow_html=True
)