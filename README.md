# AutoEDA: Automated Exploratory Data Analysis

[![PyPI version](https://badge.fury.io/py/auto-eda.svg)](https://badge.fury.io/py/auto-eda)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

AutoEDA is a comprehensive Python package for automated exploratory data analysis that helps data scientists and machine learning engineers quickly understand their datasets. It provides intelligent data type inference, detailed statistical analysis, and insightful visualizations for both structured and unstructured data.

## Features

- **Intelligent Data Type Inference:** Automatically detects and converts data types including numeric, categorical, datetime, boolean, and ID columns.
- **Comprehensive Data Analysis:** Performs analysis of distributions, correlations, outliers, and patterns.
- **Text Data Analysis:** Extracts insights from unstructured text data with word frequency, sentiment analysis, and entity extraction.
- **Enhanced Visualizations:** Creates informative plots (histograms, scatterplots, correlation matrices, word clouds, etc.).
- **Interactive Reports:** Generates detailed HTML reports with interactive visualizations.
- **Flexible Configuration:** Supports YAML/JSON configuration files.

## Project Structure

```
auto_eda/
│
├── README.md                  # Project documentation
├── setup.py                   # Package installation script
├── pyproject.toml             # PEP 517/518 build specification
├── MANIFEST.in                # Package data inclusion rules
├── LICENSE                    # License file
├── .gitignore                 # Git ignore file
│
├── auto_eda/                  # Main package directory
│   ├── __init__.py            # Package initialization
│   ├── cli.py                 # Command-line interface
│   ├── core.py                # Core EDA functionality
│   ├── data_loader.py         # Data loading utilities
│   ├── type_inference.py      # Type inference functionality
│   ├── analyzers/             # Analysis modules
│   │   ├── __init__.py
│   │   ├── numeric.py         # Numeric data analysis
│   │   ├── categorical.py     # Categorical analysis
│   │   ├── text.py            # Text analysis
│   │   ├── datetime.py        # Datetime analysis
│   │   └── correlation.py     # Correlation analysis
│   │
│   ├── visualizers/           # Visualization modules
│   │   ├── __init__.py
│   │   ├── base.py            # Base visualization class
│   │   ├── distribution.py    # Distribution plots
│   │   ├── relationship.py    # Relationship plots and scatterplots
│   │   ├── categorical.py     # Categorical plots
│   │   ├── text.py            # Text visualizations (e.g., word clouds)
│   │   └── interactive.py     # Interactive visualizations (e.g., using Plotly)
│   │
│   ├── reporting/             # Reporting modules
│   │   ├── __init__.py
│   │   ├── html_report.py     # HTML report generation
│   │   ├── markdown_report.py # Markdown report generation
│   │   ├── utils.py           # Utility functions for report generation
│   │   └── templates/         # Report templates
│   │       ├── base.html
│   │       ├── sections/
│   │       └── assets/
│   │
│   └── utils/                 # Utility functions
│       ├── __init__.py
│       ├── config.py          # Configuration handling
│       ├── logger.py          # Logging utilities
│       └── helpers.py         # Helper functions
│
├── examples/                  # Example notebooks and scripts
│   ├── basic_usage.py
│   ├── advanced_usage.py
│   ├── text_analysis.py
│
└── tests/                     # Unit tests
    ├── __init__.py
    ├── conftest.py
    ├── test_core.py
    ├── test_type_inference.py
    ├── test_analyzers/
    │   ├── __init__.py
    │   ├── test_numeric.py
    │   ├── test_categorical.py
    │   └── test_text.py
    ├── test_visualizers/
    │   ├── __init__.py
    │   ├── test_distribution.py
    │   └── test_relationship.py
    └── fixtures/              # Test data
```

## Installation

The recommended installation is via setup.py.

```bash
# Install from PyPI
pip install auto-eda

# Or install from source:
git clone https://github.com/edwardcjohnson/auto-eda.git
cd auto-eda
pip install -e .                # Install in development mode
pip install -e ".[full]"         # Install with all optional dependencies
pip install -e ".[dev]"          # Install with development dependencies
```

## Quick Start

```python
from auto_eda import AutoEDA

# Initialize with the default configuration
eda = AutoEDA(report_dir="./reports")

# Load data with intelligent type inference
eda.load_data("path/to/your/data.csv", infer_types=True)

# Run comprehensive analysis
eda.analyze()

# Generate visualizations
eda.visualize()

# Create an HTML report
eda.generate_report(title="My Data Analysis Report")
```

## Configuration

AutoEDA is customizable via YAML or JSON files. A sample configuration is provided below:

```yaml
sampling:
  enabled: true
  max_rows: 100000
  random_state: 42

type_inference:
  categorical_threshold: 0.1
  numeric_detection_strictness: "medium"
  date_inference: true
  id_detection: true

analysis:
  correlation_method: "pearson"
  outlier_detection:
    enabled: true
    method: "iqr"
    threshold: 1.5
  text_analysis:
    max_features: 100
    min_df: 2
    ngram_range: [1, 2]

visualization:
  style: "whitegrid"
  palette: "viridis"
  figure_size: [10, 6]
  dpi: 100
  interactive: true

reporting:
  include_code: true
  include_recommendations: true
  max_rows_in_report: 20
```

## Command Line Interface

AutoEDA includes a CLI for ease of use. Once installed, you can run:

```bash
# Analyze a CSV file and generate a report
auto-eda analyze path/to/data.csv --report-dir ./reports

# Use a custom configuration file
auto-eda analyze path/to/data.csv --config my_config.yaml

# Display help message
auto-eda --help
```

## Examples

The `examples/` directory includes several Python scripts that demonstrate how to use AutoEDA:

- **basic_analysis.py**  
  Creates a sample DataFrame with 20 rows and 8 columns (numeric, categorical, datetime, boolean, text, and ID) and runs a basic analysis, visualization, and HTML/Markdown report generation.

- **advanced_usage.py**  
  Demonstrates advanced usage with custom configuration and type overrides, generating both HTML and Markdown reports.

- **text_analysis.py**  
  Demonstrates text analysis on a DataFrame’s text column. This example uses the built-in functions from `auto_eda.analyzers.text` to compute word frequencies, length statistics, and sentiment scores for product reviews.

To run an example, simply execute the script from the command line. For example:

```bash
python examples/text_analysis.py
```

## Dependencies

### Core Dependencies
- Python 3.9+
- pandas>=2.0.0
- numpy>=1.24.0
- matplotlib>=3.7.0
- seaborn>=0.12.0
- plotly>=5.14.0
- scikit-learn>=1.2.0
- wordcloud>=1.9.0
- tqdm>=4.65.0
- pyyaml>=6.0.0
- jinja2>=3.1.0
- ipython>=8.12.0
- pyarrow>=12.0.0
- openpyxl>=3.1.0
- statsmodels>=0.14.0

### Optional Dependencies
- polars>=0.18.0 (for faster data processing)
- nltk>=3.8.0 (for text analysis)
- spacy>=3.5.0 (for advanced text analysis)

### Development Dependencies
- pytest>=7.3.0
- black>=23.3.0
- isort>=5.12.0
- mypy>=1.2.0
- flake8>=6.0.0
- sphinx>=6.1.0
- sphinx-rtd-theme>=1.2.0
- pytest-cov>=4.1.0

## Development Environment Setup

Clone the repository and install in development mode:

```bash
git clone https://github.com/yourusername/auto-eda.git
cd auto-eda
pip install -e ".[full,dev]"
```

To run tests:

```bash
# Install the package in development mode first
pip install -e .

# Run the tests
pytest

# Run with coverage
pytest --cov=auto_eda
```

For code formatting:

```bash
black auto_eda tests
isort auto_eda tests
```

## Documentation

Visit [https://auto-eda.readthedocs.io/](https://auto-eda.readthedocs.io/) for detailed documentation.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository,
2. Create your feature branch (`git checkout -b feature/amazing-feature`),
3. Commit your changes (`git commit -m 'Add some amazing feature'`),
4. Push to the branch (`git push origin feature/amazing-feature`),
5. Open a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.