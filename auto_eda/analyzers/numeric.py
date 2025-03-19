"""
Numeric data analysis functionality.

This module provides functions to analyze numeric columns in a DataFrame.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


def analyze_numeric_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze numeric columns in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze
    columns : List[str], optional
        List of numeric column names to analyze, by default None (all numeric columns)
    **kwargs
        Additional analysis parameters
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping column names to their analysis results
    """
    # If no columns are specified, use all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    # Filter out columns that don't exist in the DataFrame
    columns = [col for col in columns if col in df.columns]
    
    # Initialize results dictionary
    results = {}
    
    # Analyze each numeric column
    for column in columns:
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        
        # Get column data without NaN values
        column_data = df[column].dropna()
        
        if len(column_data) == 0:
            results[column] = {
                "error": "No valid data for analysis"
            }
            continue
        
        # Calculate basic statistics
        stats = column_data.describe()
        
        # Convert to dictionary and add additional metrics
        column_stats = stats.to_dict()
        
        # Add skewness and kurtosis if there are enough data points
        if len(column_data) >= 3:
            column_stats["skewness"] = column_data.skew()
            column_stats["kurtosis"] = column_data.kurtosis()
        
        # Add number of zeros and missing values
        column_stats["zeros_count"] = (df[column] == 0).sum()
        column_stats["zeros_percentage"] = (df[column] == 0).mean() * 100
        column_stats["missing_count"] = df[column].isna().sum()
        column_stats["missing_percentage"] = df[column].isna().mean() * 100
        
        # Store results
        results[column] = column_stats
    
    return results