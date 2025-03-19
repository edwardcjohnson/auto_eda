"""
Analysis modules for different data types.

This package contains modules for analyzing various types of data:
- Numeric data analysis
- Categorical data analysis
- Text data analysis
- Datetime data analysis
- Correlation analysis
"""

from auto_eda.analyzers.numeric import analyze_numeric_columns
from auto_eda.analyzers.categorical import analyze_categorical_columns
from auto_eda.analyzers.text import analyze_text_columns
from auto_eda.analyzers.datetime import analyze_datetime_columns
from auto_eda.analyzers.correlation import compute_correlations

__all__ = [
    "analyze_numeric_columns",
    "analyze_categorical_columns",
    "analyze_text_columns",
    "analyze_datetime_columns",
    "compute_correlations",
]