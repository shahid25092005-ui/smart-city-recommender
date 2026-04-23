#!/bin/bash

# SmartCity Recommender - Quick Start Script

echo "========================================="
echo "SmartRec - Smart City Recommendation Engine"
echo "========================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r app/requirements.txt

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p outputs models logs

# Run data validation
echo "🔍 Validating data files..."
python3 -c "import pandas as pd; pd.read_csv('data/smart_city_items.csv'); print('✓ Data validation passed')"

# Run tests
echo "🧪 Running unit tests..."
pytest tests/ -v --tb=short

# Check test results
if [ $? -eq 0 ]; then
    echo "✓ All tests passed!"
else
    echo "⚠️ Some tests failed. Continuing anyway..."
fi

# Generate initial outputs
echo "📊 Generating initial similarity matrix..."
python3 -c "
from src.data_loader import SmartCityDataLoader
from src.recommender import SmartRecommender
loader = SmartCityDataLoader()
recommender = SmartRecommender(loader)
recommender.initialize()
recommender.compute_similarity_matrix()
import pandas as pd
similarity_df = pd.DataFrame(recommender.similarity_matrix, 
                            index=recommender.item_names, 
                            columns=recommender.item_names)
similarity_df.to_csv('outputs/similarity_matrix.csv')
print('✓ Similarity matrix saved to outputs/similarity_matrix.csv')
"

echo ""
echo "========================================="
echo "✅ Setup complete!"
echo "========================================="
echo ""
echo "To start the Streamlit app, run:"
echo "  streamlit run app/streamlit_app.py"
echo ""
echo "Or explore the Jupyter notebook:"
echo "  jupyter notebook notebooks/01_recommendation_engine.ipynb"
echo ""
echo "To run tests again:"
echo "  pytest tests/ -v"
echo ""
echo "To deactivate virtual environment:"
echo "  deactivate"
echo ""

# Ask user if they want to start the app
read -p "Do you want to start the Streamlit app now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "🚀 Starting Streamlit app..."
    streamlit run app/streamlit_app.py
fi