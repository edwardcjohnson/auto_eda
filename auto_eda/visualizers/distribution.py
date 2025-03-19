"""
Distribution visualization functionality.

This module provides functions to create distribution plots for numeric data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

from auto_eda.visualizers.base import BaseVisualizer


def create_distribution_plots(
    df: pd.DataFrame,
    numeric_columns: List[str],
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Create distribution plots for numeric columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data
    numeric_columns : List[str]
        List of numeric column names to visualize
    output_dir : str or Path, optional
        Directory to save plots, by default None
    **kwargs
        Additional visualization parameters
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping column names to their visualization results
    """
    visualizer = BaseVisualizer(**kwargs)
    results = {}
    
    for column in numeric_columns:
        if column not in df.columns:
            continue
        
        # Get column data without NaN values
        column_data = df[column].dropna()
        
        if len(column_data) == 0:
            results[column] = {"error": "No valid data for visualization"}
            continue
        
        # Create figure with 2 subplots (histogram and boxplot)
        fig, axes = visualizer.create_figure(nrows=1, ncols=2, figsize=(12, 5))
        
        # Histogram with KDE
        sns.histplot(column_data, kde=True, ax=axes[0])
        axes[0].set_title(f"Distribution of {column}")
        axes[0].set_xlabel(column)
        axes[0].set_ylabel("Frequency")
        
        # Boxplot
        sns.boxplot(x=column_data, ax=axes[1])
        axes[1].set_title(f"Boxplot of {column}")
        axes[1].set_xlabel(column)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if output directory is provided
        saved_files = {}
        if output_dir is not None:
            saved_files = visualizer.save_figure(
                fig, f"distribution_{column}", directory=output_dir
            )
        
        # Close figure to free memory
        visualizer.close_figure(fig)
        
        # Store results
        results[column] = {
            "plot_type": "distribution",
            "saved_files": saved_files
        }
    
    return results


def create_histogram(
    df: pd.DataFrame,
    column: str,
    bins: Optional[int] = None,
    kde: bool = True,
    **kwargs
) -> plt.Figure:
    """
    Create a histogram for a numeric column.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data
    column : str
        The column name to visualize
    bins : int, optional
        Number of bins, by default None (auto-determined)
    kde : bool, optional
        Whether to include KDE, by default True
    **kwargs
        Additional visualization parameters
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    visualizer = BaseVisualizer(**kwargs)
    
    # Create figure
    fig, ax = visualizer.create_figure()
    
    # Create histogram
    sns.histplot(df[column].dropna(), bins=bins, kde=kde, ax=ax)
    
    # Set title and labels
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    
    return fig


def create_boxplot(
    df: pd.DataFrame,
    column: str,
    **kwargs
) -> plt.Figure:
    """
    Create a boxplot for a numeric column.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data
    column : str
        The column name to visualize
    **kwargs
        Additional visualization parameters
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    visualizer = BaseVisualizer(**kwargs)
    
    # Create figure
    fig, ax = visualizer.create_figure()
    
    # Create boxplot
    sns.boxplot(x=df[column].dropna(), ax=ax)
    
    # Set title and labels
    ax.set_title(f"Boxplot of {column}")
    ax.set_xlabel(column)
    
    return fig