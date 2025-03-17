"""
Type inference utilities.

This module provides functions to infer column types in a DataFrame.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import re
from datetime import datetime


def infer_column_types(
    df: pd.DataFrame,
    **kwargs
) -> Dict[str, List[str]]:
    """
    Infer the types of columns in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze
    **kwargs
        Additional inference parameters
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping type categories to lists of column names
    """
    # Initialize column type categories
    column_types = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "text": [],
        "boolean": [],
        "other": []
    }
    
    # Analyze each column
    for column in df.columns:
        # Get column data
        column_data = df[column]
        
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(column_data):
            # Check if it's likely a categorical variable encoded as numeric
            if column_data.nunique() < 10 and column_data.nunique() / len(column_data) < 0.05:
                column_types["categorical"].append(column)
            else:
                column_types["numeric"].append(column)
            continue
        
        # Check if column is datetime
        if pd.api.types.is_datetime64_dtype(column_data):
            column_types["datetime"].append(column)
            continue
        
        # Check if column is boolean
        if pd.api.types.is_bool_dtype(column_data):
            column_types["boolean"].append(column)
            column_types["categorical"].append(column)  # Boolean can also be treated as categorical
            continue
        
        # For object dtype, try to infer more specific types
        if pd.api.types.is_object_dtype(column_data):
            # Sample non-null values for type inference
            sample = column_data.dropna().sample(min(100, len(column_data.dropna()))).tolist()
            
            if not sample:
                column_types["other"].append(column)
                continue
            
            # Try to convert to datetime
            try:
                pd.to_datetime(sample)
                column_types["datetime"].append(column)
                continue
            except (ValueError, TypeError):
                pass
            
            # Check if all values are strings
            if all(isinstance(x, str) for x in sample):
                # Check if it's likely a categorical variable
                if column_data.nunique() < 10 or column_data.nunique() / len(column_data) < 0.05:
                    column_types["categorical"].append(column)
                # Check if it's likely a text variable
                elif column_data.str.len().mean() > 50:
                    column_types["text"].append(column)
                else:
                    column_types["categorical"].append(column)
                continue
            
            # Check if all values are boolean-like
            bool_values = ['true', 'false', 'yes', 'no', 'y', 'n', 't', 'f', '1', '0']
            if all(str(x).lower() in bool_values for x in sample):
                column_types["boolean"].append(column)
                column_types["categorical"].append(column)
                continue
            
            # Default to categorical
            column_types["categorical"].append(column)
        else:
            # Default to other for unknown types
            column_types["other"].append(column)
    
    return column_types


def detect_numeric_columns(
    df: pd.DataFrame,
    **kwargs
) -> List[str]:
    """
    Detect numeric columns in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze
    **kwargs
        Additional detection parameters
        
    Returns
    -------
    List[str]
        List of numeric column names
    """
    column_types = infer_column_types(df, **kwargs)
    return column_types["numeric"]


def detect_categorical_columns(
    df: pd.DataFrame,
    **kwargs
) -> List[str]:
    """
    Detect categorical columns in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze
    **kwargs
        Additional detection parameters
        
    Returns
    -------
    List[str]
        List of categorical column names
    """
    column_types = infer_column_types(df, **kwargs)
    return column_types["categorical"]


def detect_datetime_columns(
    df: pd.DataFrame,
    **kwargs
) -> List[str]:
    """
    Detect datetime columns in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze
    **kwargs
        Additional detection parameters
        
    Returns
    -------
    List[str]
        List of datetime column names
    """
    column_types = infer_column_types(df, **kwargs)
    return column_types["datetime"]


def detect_text_columns(
    df: pd.DataFrame,
    **kwargs
) -> List[str]:
    """
    Detect text columns in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze
    **kwargs
        Additional detection parameters
        
    Returns
    -------
    List[str]
        List of text column names
    """
    column_types = infer_column_types(df, **kwargs)
    return column_types["text"]


def detect_boolean_columns(
    df: pd.DataFrame,
    **kwargs
) -> List[str]:
    """
    Detect boolean columns in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze
    **kwargs
        Additional detection parameters
        
    Returns
    -------
    List[str]
        List of boolean column names
    """
    column_types = infer_column_types(df, **kwargs)
    return column_types["boolean"]


def is_numeric_column(
    series: pd.Series,
    **kwargs
) -> bool:
    """
    Check if a series is numeric.
    
    Parameters
    ----------
    series : pd.Series
        The series to check
    **kwargs
        Additional check parameters
        
    Returns
    -------
    bool
        True if the series is numeric, False otherwise
    """
    return pd.api.types.is_numeric_dtype(series)


def is_categorical_column(
    series: pd.Series,
    max_unique_ratio: float = 0.05,
    max_unique_values: int = 10,
    **kwargs
) -> bool:
    """
    Check if a series is categorical.
    
    Parameters
    ----------
    series : pd.Series
        The series to check
    max_unique_ratio : float, optional
        Maximum ratio of unique values to total values, by default 0.05
    max_unique_values : int, optional
        Maximum number of unique values, by default 10
    **kwargs
        Additional check parameters
        
    Returns
    -------
    bool
        True if the series is categorical, False otherwise
    """
    # If it's already a categorical dtype, return True
    if pd.api.types.is_categorical_dtype(series):
        return True
    
    # If it's a boolean dtype, return True
    if pd.api.types.is_bool_dtype(series):
        return True
    
    # Check number of unique values
    n_unique = series.nunique()
    n_total = len(series)
    
    # If there are too many unique values, it's not categorical
    if n_unique > max_unique_values and n_unique / n_total > max_unique_ratio:
        return False
    
    return True


def is_datetime_column(
    series: pd.Series,
    **kwargs
) -> bool:
    """
    Check if a series is datetime.
    
    Parameters
    ----------
    series : pd.Series
        The series to check
    **kwargs
        Additional check parameters
        
    Returns
    -------
    bool
        True if the series is datetime, False otherwise
    """
    # If it's already a datetime dtype, return True
    if pd.api.types.is_datetime64_dtype(series):
        return True
    
    # If it's not an object dtype, return False
    if not pd.api.types.is_object_dtype(series):
        return False
    
    # Try to convert to datetime
    try:
        pd.to_datetime(series.dropna().head(100))
        return True
    except (ValueError, TypeError):
        return False


def is_text_column(
    series: pd.Series,
    min_mean_length: int = 50,
    **kwargs
) -> bool:
    """
    Check if a series is text.
    
    Parameters
    ----------
    series : pd.Series
        The series to check
    min_mean_length : int, optional
        Minimum mean length of strings to be considered text, by default 50
    **kwargs
        Additional check parameters
        
    Returns
    -------
    bool
        True if the series is text, False otherwise
    """
    # If it's not an object dtype, return False
    if not pd.api.types.is_object_dtype(series):
        return False
    
    # Check if all values are strings
    sample = series.dropna().head(100)
    if not all(isinstance(x, str) for x in sample):
        return False
    
    # Check mean length of strings
    mean_length = sample.str.len().mean()
    return mean_length >= min_mean_length


def is_boolean_column(
    series: pd.Series,
    **kwargs
) -> bool:
    """
    Check if a series is boolean.
    
    Parameters
    ----------
    series : pd.Series
        The series to check
    **kwargs
        Additional check parameters
        
    Returns
    -------
    bool
        True if the series is boolean, False otherwise
    """
    # If it's already a boolean dtype, return True
    if pd.api.types.is_bool_dtype(series):
        return True
    
    # Check if all values are boolean-like
    sample = series.dropna().head(100)
    bool_values = ['true', 'false', 'yes', 'no', 'y', 'n', 't', 'f', '1', '0']
    return all(str(x).lower() in bool_values for x in sample)