"""
Categorical visualization functionality.

This module provides functions to create visualizations for categorical data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

from auto_eda.visualizers.base import BaseVisualizer


def create_categorical_plots(
    df: pd.DataFrame,
    categorical_columns: List[str],
    output_dir: Optional[Union[str, Path]] = None,
    max_categories: int = 20,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Create plots for categorical columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data
    categorical_columns : List[str]
        List of categorical column names to visualize
    output_dir : str or Path, optional
        Directory to save plots, by default None
    max_categories : int, optional
        Maximum number of categories to display, by default 20
    **kwargs
        Additional visualization parameters
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping column names to their visualization results
    """
    visualizer = BaseVisualizer(**kwargs)
    results = {}
    
    for column in categorical_columns:
        if column not in df.columns:
            continue
        
        # Get column data without NaN values
        column_data = df[column].dropna()
        
        if len(column_data) == 0:
            results[column] = {"error": "No valid data for visualization"}
            continue
        
        # Get value counts
        value_counts = column_data.value_counts()
        
        # If there are too many categories, limit to top categories
        if len(value_counts) > max_categories:
            top_categories = value_counts.nlargest(max_categories - 1)
            other_count = value_counts.sum() - top_categories.sum()
            
            # Add "Other" category
            value_counts = pd.concat([top_categories, pd.Series([other_count], index=["Other"])])
        
        # Create figure
        fig, ax = visualizer.create_figure(figsize=(12, 6))
        
        # Create bar plot
        sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
        
        # Set title and labels
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        
        # Rotate x-axis labels if there are many categories
        if len(value_counts) > 5:
            plt.xticks(rotation=45, ha="right")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if output directory is provided
        saved_files = {}
        if output_dir is not None:
            saved_files = visualizer.save_figure(
                fig, f"categorical_{column}", directory=output_dir
            )
        
        # Close figure to free memory
        visualizer.close_figure(fig)
        
        # Store results
        results[column] = {
            "plot_type": "bar",
            "saved_files": saved_files
        }
    
    return results


def create_bar_plot(
    df: pd.DataFrame,
    column: str,
    max_categories: int = 20,
    **kwargs
) -> plt.Figure:
    """
    Create a bar plot for a categorical column.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data
    column : str
        The column name to visualize
    max_categories : int, optional
        Maximum number of categories to display, by default 20
    **kwargs
        Additional visualization parameters
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    visualizer = BaseVisualizer(**kwargs)
    
    # Get column data without NaN values
    column_data = df[column].dropna()
    
    # Get value counts
    value_counts = column_data.value_counts()
    
    # If there are too many categories, limit to top categories
    if len(value_counts) > max_categories:
        top_categories = value_counts.nlargest(max_categories - 1)
        other_count = value_counts.sum() - top_categories.sum()
        
        # Add "Other" category
        value_counts = pd.concat([top_categories, pd.Series([other_count], index=["Other"])])
    
    # Create figure
    fig, ax = visualizer.create_figure()
    
    # Create bar plot
    sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
    
    # Set title and labels
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    
    # Rotate x-axis labels if there are many categories
    if len(value_counts) > 5:
        plt.xticks(rotation=45, ha="right")
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def create_pie_chart(
    df: pd.DataFrame,
    column: str,
    max_categories: int = 10,
    **kwargs
) -> plt.Figure:
    """
    Create a pie chart for a categorical column.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data
    column : str
        The column name to visualize
    max_categories : int, optional
        Maximum number of categories to display, by default 10
    **kwargs
        Additional visualization parameters
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    visualizer = BaseVisualizer(**kwargs)
    
    # Get column data without NaN values
    column_data = df[column].dropna()
    
    # Get value counts
    value_counts = column_data.value_counts()
    
    # If there are too many categories, limit to top categories
    if len(value_counts) > max_categories:
        top_categories = value_counts.nlargest(max_categories - 1)
        other_count = value_counts.sum() - top_categories.sum()
        
        # Add "Other" category
        value_counts = pd.concat([top_categories, pd.Series([other_count], index=["Other"])])
    
    # Create figure
    fig, ax = visualizer.create_figure()
    
    # Create pie chart
    ax.pie(
        value_counts.values,
        labels=value_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        shadow=False,
    )
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis("equal")
    
    # Set title
    ax.set_title(f"Distribution of {column}")
    
    return fig


def create_count_plot(
    df: pd.DataFrame,
    column: str,
    hue: Optional[str] = None,
    max_categories: int = 20,
    **kwargs
) -> plt.Figure:
    """
    Create a count plot for a categorical column, optionally grouped by another column.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data
    column : str
        The column name to visualize
    hue : str, optional
        Column name to group by, by default None
    max_categories : int, optional
        Maximum number of categories to display, by default 20
    **kwargs
        Additional visualization parameters
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    visualizer = BaseVisualizer(**kwargs)
    
    # If there are too many categories, limit the data
    if df[column].nunique() > max_categories:
        # Get top categories
        top_categories = df[column].value_counts().nlargest(max_categories - 1).index.tolist()
        
        # Filter data to include only top categories and "Other"
        filtered_df = df.copy()
        filtered_df.loc[~filtered_df[column].isin(top_categories), column] = "Other"
    else:
        filtered_df = df
    
    # Create figure
    fig, ax = visualizer.create_figure(figsize=(12, 6))
    
    # Create count plot
    sns.countplot(x=column, hue=hue, data=filtered_df, ax=ax)
    
    # Set title and labels
    ax.set_title(f"Count of {column}" + (f" by {hue}" if hue else ""))
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    
    # Rotate x-axis labels if there are many categories
    if filtered_df[column].nunique() > 5:
        plt.xticks(rotation=45, ha="right")
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def create_categorical_heatmap(
    df: pd.DataFrame,
    column1: str,
    column2: str,
    normalize: bool = True,
    **kwargs
) -> plt.Figure:
    """
    Create a heatmap showing the relationship between two categorical columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data
    column1 : str
        First categorical column name
    column2 : str
        Second categorical column name
    normalize : bool, optional
        Whether to normalize the counts, by default True
    **kwargs
        Additional visualization parameters
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    visualizer = BaseVisualizer(**kwargs)
    
    # Create contingency table
    if normalize:
        contingency = pd.crosstab(df[column1], df[column2], normalize="index")
        fmt = ".1%"
        vmin, vmax = 0, 1
    else:
        contingency = pd.crosstab(df[column1], df[column2])
        fmt = "d"
        vmin, vmax = None, None
    
    # Create figure
    fig, ax = visualizer.create_figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        contingency,
        annot=True,
        fmt=fmt,
        cmap="YlGnBu",
        vmin=vmin,
        vmax=vmax,
        ax=ax
    )
    
    # Set title and labels
    ax.set_title(f"Relationship between {column1} and {column2}")
    ax.set_xlabel(column2)
    ax.set_ylabel(column1)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig