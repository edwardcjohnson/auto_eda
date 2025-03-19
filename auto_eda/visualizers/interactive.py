"""
Interactive visualization functionality.

This module provides functions to create interactive visualizations using Plotly.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import json
import os

# Check if plotly is available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def create_interactive_plots(
    df: pd.DataFrame,
    numeric_columns: List[str],
    categorical_columns: List[str],
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Create interactive plots for the DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data
    numeric_columns : List[str]
        List of numeric column names
    categorical_columns : List[str]
        List of categorical column names
    output_dir : str or Path, optional
        Directory to save plots, by default None
    **kwargs
        Additional visualization parameters
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping plot types to their visualization results
    """
    if not PLOTLY_AVAILABLE:
        return {"error": "Plotly is not installed. Install it with: pip install plotly"}
    
    results = {}
    
    # Create output directory if provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create scatter plot matrix for numeric columns
    if len(numeric_columns) >= 2:
        try:
            scatter_matrix_result = create_scatter_matrix(
                df, numeric_columns, output_dir=output_dir, **kwargs
            )
            results["scatter_matrix"] = scatter_matrix_result
        except Exception as e:
            results["scatter_matrix"] = {"error": f"Error creating scatter matrix: {str(e)}"}
    
    # Create bar plots for categorical columns
    for column in categorical_columns:
        if column not in df.columns:
            continue
        
        try:
            bar_result = create_interactive_bar(
                df, column, output_dir=output_dir, **kwargs
            )
            results[f"bar_{column}"] = bar_result
        except Exception as e:
            results[f"bar_{column}"] = {"error": f"Error creating bar plot: {str(e)}"}
    
    # Create histograms for numeric columns
    for column in numeric_columns:
        if column not in df.columns:
            continue
        
        try:
            hist_result = create_interactive_histogram(
                df, column, output_dir=output_dir, **kwargs
            )
            results[f"histogram_{column}"] = hist_result
        except Exception as e:
            results[f"histogram_{column}"] = {"error": f"Error creating histogram: {str(e)}"}
    
    # Create box plots for numeric columns
    for column in numeric_columns:
        if column not in df.columns:
            continue
        
        try:
            box_result = create_interactive_box(
                df, column, output_dir=output_dir, **kwargs
            )
            results[f"box_{column}"] = box_result
        except Exception as e:
            results[f"box_{column}"] = {"error": f"Error creating box plot: {str(e)}"}
    
    return results


def create_scatter_matrix(
    df: pd.DataFrame,
    columns: List[str],
    color_column: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create an interactive scatter plot matrix.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data
    columns : List[str]
        List of column names to include in the scatter matrix
    color_column : str, optional
        Column name to use for coloring points, by default None
    output_dir : str or Path, optional
        Directory to save the plot, by default None
    **kwargs
        Additional visualization parameters
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with visualization results
    """
    if not PLOTLY_AVAILABLE:
        return {"error": "Plotly is not installed. Install it with: pip install plotly"}