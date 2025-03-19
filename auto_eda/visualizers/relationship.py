"""
Relationship visualization functionality.

This module provides functions to create relationship plots between variables.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

from auto_eda.visualizers.base import BaseVisualizer


def create_relationship_plots(
    df: pd.DataFrame,
    numeric_columns: List[str],
    output_dir: Optional[Union[str, Path]] = None,
    max_plots: int = 20,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Create relationship plots between numeric columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data
    numeric_columns : List[str]
        List of numeric column names to visualize
    output_dir : str or Path, optional
        Directory to save plots, by default None
    max_plots : int, optional
        Maximum number of plots to create, by default 20
    **kwargs
        Additional visualization parameters
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping column pairs to their visualization results
    """
    visualizer = BaseVisualizer(**kwargs)
    results = {}
    
    # Filter columns that exist in the DataFrame
    numeric_columns = [col for col in numeric_columns if col in df.columns]
    
    # If there are too many columns, limit the number of plots
    if len(numeric_columns) > 5:
        import itertools
        column_pairs = list(itertools.combinations(numeric_columns, 2))
        
        # If there are too many pairs, select a subset
        if len(column_pairs) > max_plots:
            import random
            random.seed(42)  # For reproducibility
            column_pairs = random.sample(column_pairs, max_plots)
    else:
        # Create a correlation matrix plot for all columns
        fig, ax = visualizer.create_figure(figsize=(10, 8))
        
        # Create correlation heatmap
        corr_matrix = df[numeric_columns].corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
        
        ax.set_title("Correlation Matrix")
        
        # Save figure if output directory is provided
        saved_files = {}
        if output_dir is not None:
            saved_files = visualizer.save_figure(
                fig, "correlation_matrix", directory=output_dir
            )
        
        # Close figure to free memory
        visualizer.close_figure(fig)
        
        # Store results
        results["correlation_matrix"] = {
            "plot_type": "heatmap",
            "saved_files": saved_files
        }
        
        # Create scatterplots for each pair
        column_pairs = [(col1, col2) for col1 in numeric_columns for col2 in numeric_columns if col1 < col2]
    
    # Create scatterplots for column pairs
    for col1, col2 in column_pairs:
        # Get data without NaN values
        valid_data = df[[col1, col2]].dropna()
        
        if len(valid_data) < 2:
            results[f"{col1}_vs_{col2}"] = {"error": "Not enough valid data for visualization"}
            continue
        
        # Create figure
        fig, ax = visualizer.create_figure()
        
        # Create scatterplot with regression line
        sns.regplot(x=col1, y=col2, data=valid_data, ax=ax)
        
        # Set title and labels
        ax.set_title(f"{col1} vs {col2}")
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        
        # Add correlation coefficient to the plot
        corr = valid_data[col1].corr(valid_data[col2])
        ax.annotate(f"Correlation: {corr:.2f}", xy=(0.05, 0.95), xycoords="axes fraction")
        
        # Save figure if output directory is provided
        saved_files = {}
        if output_dir is not None:
            saved_files = visualizer.save_figure(
                fig, f"scatter_{col1}_vs_{col2}", directory=output_dir
            )
        
        # Close figure to free memory
        visualizer.close_figure(fig)
        
        # Store results
        results[f"{col1}_vs_{col2}"] = {
            "plot_type": "scatter",
            "correlation": corr,
            "saved_files": saved_files
        }
    
    return results


def create_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "pearson",
    **kwargs
) -> plt.Figure:
    """
    Create a correlation heatmap.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data
    columns : List[str], optional
        List of column names to include, by default None (uses all numeric columns)
    method : str, optional
        Correlation method, by default "pearson"
    **kwargs
        Additional visualization parameters
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    visualizer = BaseVisualizer(**kwargs)
    
    # If no columns are specified, use all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=["number"]).columns.tolist()
    
    # Create figure