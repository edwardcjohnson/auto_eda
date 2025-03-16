# AutoEDA: Automated Exploratory Data Analysis

[![PyPI version](https://badge.fury.io/py/auto-eda.svg)](https://badge.fury.io/py/auto-eda)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

AutoEDA is a comprehensive Python package for automated exploratory data analysis that helps data scientists and analysts quickly understand their datasets. It provides intelligent data type inference, detailed statistical analysis, and insightful visualizations for both structured and unstructured data.

## Features

- **Intelligent Data Type Inference**: Automatically detects and converts data types including numeric, categorical, datetime, boolean, and ID columns
- **Comprehensive Data Analysis**: Analyzes distributions, correlations, outliers, and patterns in your data
- **Text Data Analysis**: Extracts insights from unstructured text data with word frequency, sentiment analysis, and entity extraction
- **Enhanced Visualizations**: Creates informative plots including histograms, scatterplots, correlation matrices, and word clouds
- **Interactive Reports**: Generates detailed HTML reports with interactive visualizations
- **Flexible Configuration**: Customizable analysis parameters through YAML/JSON configuration files

## Installation

```bash
# Install from PyPI
pip install auto-eda

# Or install from source
git clone https://github.com/edwardcjohnson/auto-eda.git
cd auto-eda
pip install -e .