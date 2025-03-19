"""
Visualization modules for different data types and relationships.

This package contains modules for creating various types of visualizations:
- Distribution plots for numeric data
- Relationship plots for exploring correlations
- Categorical plots for categorical data
- Text visualizations like word clouds
- Interactive visualizations using Plotly
"""

from auto_eda.visualizers.base import BaseVisualizer
from auto_eda.visualizers.distribution import create_distribution_plots
from auto_eda.visualizers.relationship import create_relationship_plots
from auto_eda.visualizers.categorical import create_categorical_plots
from auto_eda.visualizers.text import create_text_visualizations
from auto_eda.visualizers.interactive import create_interactive_plots

__all__ = [
    "BaseVisualizer",
    "create_distribution_plots",
    "create_relationship_plots",
    "create_categorical_plots",
    "create_text_visualizations",
    "create_interactive_plots",
]