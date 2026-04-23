# SmartRec - Context-Aware Smart City Recommendation Engine

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io)

## 🏙️ Overview

SmartRec is a production-ready recommendation engine for Smart City infrastructure, utilizing content-based filtering with TF-IDF and cosine similarity. The system provides context-aware recommendations for urban services across multiple categories including Mobility, Environment, Safety, Utility, and Civic amenities.

## ✨ Features

- **Context-Aware Recommendations**: Switch between tourist, resident, emergency, and night_mode contexts
- **Hybrid Scoring**: Combines content similarity with popularity metrics
- **Real-time Explanations**: Understand why items are recommended
- **Interactive Dashboard**: Streamlit-based web application
- **Comprehensive Testing**: Unit tests for all major components
- **Performance Optimized**: Caching and efficient similarity computation

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or extract the project**
```bash
cd smart-city-recommender
```

2. **Set up a virtual environment (Recommended)**
```bash
py -3.11 -m venv myenv
# On Windows
myenv\Scripts\activate
# On macOS/Linux
source myenv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app/app.py
```

## 🧪 Running Tests

To ensure everything is set up correctly, you can run the unit tests:
```bash
pytest tests/
```

## 📂 Project Structure

- `app/`: Streamlit dashboard application (`app.py`).
- `src/`: Core recommendation engine logic.
- `data/`: Dataset files (CSV, JSON).
- `notebooks/`: Exploratory Data Analysis and prototyping.
- `tests/`: Unit tests for the engine and data validation.
- `outputs/`: Directory for generated similarity matrices and logs.