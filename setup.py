from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="smartcity_recommender",
    version="1.0.0",
    author="SmartRec Team",
    author_email="smartrec@example.com",
    description="Context-Aware Recommendation Engine for Smart City Infrastructure",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/smartcity_recommender",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.10.0",
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "jupyter>=1.0.0",
        "pytest>=7.4.0",
    ],
    extras_require={
        "dev": [
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
    },
    package_data={
        "smartcity_recommender": [
            "data/*.csv",
            "data/*.json",
            "outputs/*",
        ],
    },
    entry_points={
        "console_scripts": [
            "smartrec=app.streamlit_app:main",
        ],
    },
)