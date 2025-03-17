"""
Text data analysis functionality.

This module provides functions to analyze text columns in a DataFrame.
"""

import pandas as pd
from typing import Dict, List, Any, Optional
import re
from collections import Counter


def analyze_text_columns(
    df: pd.DataFrame,
    columns: List[str],
    max_words: int = 100,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze text columns in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.
    columns : List[str]
        The list of text column names to analyze.
    max_words : int, optional
        Maximum number of words to include in the word frequency table, by default 100.
    **kwargs
        Additional arguments for analysis configuration.
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        A dictionary mapping column names to their analysis results.
    """
    results = {}
    
    for column in columns:
        if column not in df.columns:
            continue
        
        # Get column data without NaN values
        column_data = df[column].dropna().astype(str)
        
        # Basic statistics
        stats = {
            'count': len(column_data),
            'missing_count': df[column].isna().sum(),
            'missing_percentage': df[column].isna().mean() * 100,
            'empty_count': (column_data == '').sum(),
            'empty_percentage': ((column_data == '').sum() / len(column_data)) * 100 if len(column_data) > 0 else 0,
        }
        
        # Length statistics
        if len(column_data) > 0:
            lengths = column_data.str.len()
            stats.update({
                'avg_length': lengths.mean(),
                'min_length': lengths.min(),
                'max_length': lengths.max(),
                'median_length': lengths.median()
            })
        
        # Word count statistics
        if len(column_data) > 0:
            word_counts = column_data.str.split().str.len()
            stats.update({
                'avg_word_count': word_counts.mean(),
                'min_word_count': word_counts.min(),
                'max_word_count': word_counts.max(),
                'median_word_count': word_counts.median()
            })
        
        # Word frequency analysis
        if len(column_data) > 0:
            # Combine all text and tokenize
            all_text = ' '.join(column_data).lower()
            words = re.findall(r'\b\w+\b', all_text)
            
            # Count word frequencies
            word_freq = Counter(words)
            
            # Get the most common words
            most_common = word_freq.most_common(max_words)
            
            # Create word frequency table
            word_freq_table = pd.DataFrame(most_common, columns=['word', 'count'])
            word_freq_table['percentage'] = word_freq_table['count'] / len(words) * 100
            
            stats['word_frequency'] = word_freq_table.to_dict('records')
            stats['total_words'] = len(words)
            stats['unique_words'] = len(word_freq)
        
        results[column] = stats
    
    return results


def perform_sentiment_analysis(
    df: pd.DataFrame,
    text_columns: List[str],
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Perform sentiment analysis on text columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.
    text_columns : List[str]
        The list of text column names to analyze.
    **kwargs
        Additional arguments for sentiment analysis configuration.
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        A dictionary mapping column names to their sentiment analysis results.
    """
    results = {}
    
    # Check if nltk is available
    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
        
        # Download required NLTK resources if not already downloaded
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        
        # Initialize sentiment analyzer
        sia = SentimentIntensityAnalyzer()
    except ImportError:
        # Return empty results if NLTK is not available
        for column in text_columns:
            if column in df.columns:
                results[column] = {
                    'error': 'NLTK package is required for sentiment analysis but not installed.'
                }
        return results
    
    for column in text_columns:
        if column not in df.columns:
            continue
        
        # Get column data without NaN values
        column_data = df[column].dropna().astype(str)
        
        if len(column_data) == 0:
            results[column] = {
                'error': 'No valid text data found in column.'
            }
            continue
        
        # Perform sentiment analysis
        sentiment_scores = column