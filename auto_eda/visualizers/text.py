"""
Text visualization functionality.

This module provides functions to create visualizations for text data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import re
from collections import Counter

from auto_eda.visualizers.base import BaseVisualizer


def create_text_visualizations(
    df: pd.DataFrame,
    text_columns: List[str],
    output_dir: Optional[Union[str, Path]] = None,
    max_words: int = 100,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Create visualizations for text columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data
    text_columns : List[str]
        List of text column names to visualize
    output_dir : str or Path, optional
        Directory to save plots, by default None
    max_words : int, optional
        Maximum number of words to include in word cloud, by default 100
    **kwargs
        Additional visualization parameters
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping column names to their visualization results
    """
    visualizer = BaseVisualizer(**kwargs)
    results = {}
    
    for column in text_columns:
        if column not in df.columns:
            continue
        
        # Get column data without NaN values
        column_data = df[column].dropna().astype(str)
        
        if len(column_data) == 0:
            results[column] = {"error": "No valid data for visualization"}
            continue
        
        # Create word frequency plot
        try:
            fig_freq = create_word_frequency_plot(df, column, top_n=20, **kwargs)
            
            # Save figure if output directory is provided
            saved_files_freq = {}
            if output_dir is not None:
                saved_files_freq = visualizer.save_figure(
                    fig_freq, f"word_freq_{column}", directory=output_dir
                )
            
            # Close figure to free memory
            visualizer.close_figure(fig_freq)
            
            # Store results
            results[column] = {
                "plot_type": "word_frequency",
                "saved_files": saved_files_freq
            }
        except Exception as e:
            results[column] = {"error": f"Error creating word frequency plot: {str(e)}"}
        
        # Create word cloud
        try:
            # Check if wordcloud package is available
            try:
                from wordcloud import WordCloud
                
                fig_cloud = create_word_cloud(df, column, max_words=max_words, **kwargs)
                
                # Save figure if output directory is provided
                saved_files_cloud = {}
                if output_dir is not None:
                    saved_files_cloud = visualizer.save_figure(
                        fig_cloud, f"wordcloud_{column}", directory=output_dir
                    )
                
                # Close figure to free memory
                visualizer.close_figure(fig_cloud)
                
                # Update results
                if column in results:
                    results[column]["wordcloud_saved_files"] = saved_files_cloud
                else:
                    results[column] = {
                        "plot_type": "wordcloud",
                        "saved_files": saved_files_cloud
                    }
            except ImportError:
                if column in results:
                    results[column]["wordcloud_error"] = "WordCloud package not installed"
                else:
                    results[column] = {"error": "WordCloud package not installed"}
        except Exception as e:
            if column in results:
                results[column]["wordcloud_error"] = f"Error creating word cloud: {str(e)}"
            else:
                results[column] = {"error": f"Error creating word cloud: {str(e)}"}
    
    return results


def create_word_frequency_plot(
    df: pd.DataFrame,
    column: str,
    top_n: int = 20,
    **kwargs
) -> plt.Figure:
    """
    Create a word frequency plot for a text column.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data
    column : str
        The column name to visualize
    top_n : int, optional
        Number of top words to display, by default 20
    **kwargs
        Additional visualization parameters
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    visualizer = BaseVisualizer(**kwargs)
    
    # Get column data without NaN values
    column_data = df[column].dropna().astype(str)
    
    # Combine all text and tokenize
    all_text = ' '.join(column_data).lower()
    words = re.findall(r'\b\w+\b', all_text)
    
    # Count word frequencies
    word_freq = Counter(words)
    
    # Get the most common words
    most_common = word_freq.most_common(top_n)
    
    # Create DataFrame for plotting
    word_df = pd.DataFrame(most_common, columns=['word', 'count'])
    
    # Create figure
    fig, ax = visualizer.create_figure(figsize=(12, 6))
    
    # Create horizontal bar plot
    sns.barplot(x='count', y='word', data=word_df, ax=ax)
    
    # Set title and labels
    ax.set_title(f"Top {top_n} Words in {column}")
    ax.set_xlabel("Count")
    ax.set_ylabel("Word")
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def create_word_cloud(
    df: pd.DataFrame,
    column: str,
    max_words: int = 100,
    background_color: str = "white",
    **kwargs
) -> plt.Figure:
    """
    Create a word cloud for a text column.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data
    column : str
        The column name to visualize
    max_words : int, optional
        Maximum number of words to include, by default 100
    background_color : str, optional
        Background color of the word cloud, by default "white"
    **kwargs
        Additional visualization parameters
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        raise ImportError("WordCloud package is required for this function. Install it with: pip install wordcloud")
    
    visualizer = BaseVisualizer(**kwargs)
    
    # Get column data without NaN values
    column_data = df[column].dropna().astype(str)
    
    # Combine all text
    all_text = ' '.join(column_data).lower()
    
    # Create word cloud
    wordcloud = WordCloud(
        max_words=max_words,
        background_color=background_color,
        width=800,
        height=400,
        contour_width=1,
        contour_color='steelblue'
    ).generate(all_text)
    
    # Create figure
    fig, ax = visualizer.create_figure(figsize=(10, 6))
    
    # Display word cloud
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    # Set title
    ax.set_title(f"Word Cloud for {column}")
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def create_text_length_distribution(
    df: pd.DataFrame,
    column: str,
    bins: int = 30,
    **kwargs
) -> plt.Figure:
    """
    Create a distribution plot of text lengths for a text column.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data
    column : str
        The column name to visualize
    bins : int, optional
        Number of bins for the histogram, by default 30
    **kwargs
        Additional visualization parameters
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    visualizer = BaseVisualizer(**kwargs)
    
    # Get column data without NaN values
    column_data = df[column].dropna().astype(str)
    
    # Calculate text lengths
    text_lengths = column_data.str.len()
    
    # Create figure
    fig, ax = visualizer.create_figure()
    
    # Create histogram with KDE
    sns.histplot(text_lengths, bins=bins, kde=True, ax=ax)
    
    # Set title and labels
    ax.set_title(f"Distribution of Text Lengths in {column}")
    ax.set_xlabel("Text Length (characters)")
    ax.set_ylabel("Frequency")
    
    # Add vertical line for mean and median
    mean_length = text_lengths.mean()
    median_length = text_lengths.median()
    
    ax.axvline(mean_length, color='red', linestyle='--', label=f'Mean: {mean_length:.1f}')
    ax.axvline(median_length, color='green', linestyle='-.', label=f'Median: {median_length:.1f}')
    
    # Add legend
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def create_word_count_distribution(
    df: pd.DataFrame,
    column: str,
    bins: int = 30,
    **kwargs
) -> plt.Figure:
    """
    Create a distribution plot of word counts for a text column.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data
    column : str
        The column name to visualize
    bins : int, optional
        Number of bins for the histogram, by default 30
    **kwargs
        Additional visualization parameters
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    visualizer = BaseVisualizer(**kwargs)
    
    # Get column data without NaN values
    column_data = df[column].dropna().astype(str)
    
    # Calculate word counts
    word_counts = column_data.str.split().str.len()
    
    # Create figure
    fig, ax = visualizer.create_figure()
    
    # Create histogram with KDE
    sns.histplot(word_counts, bins=bins, kde=True, ax=ax)
    
    # Set title and labels
    ax.set_title(f"Distribution of Word Counts in {column}")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Frequency")
    
    # Add vertical line for mean and median
    mean_count = word_counts.mean()
    median_count = word_counts.median()
    
    ax.axvline(mean_count, color='red', linestyle='--', label=f'Mean: {mean_count:.1f}')
    ax.axvline(median_count, color='green', linestyle='-.', label=f'Median: {median_count:.1f}')
    
    # Add legend
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig