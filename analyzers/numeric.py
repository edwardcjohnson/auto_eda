"""
Numeric data analysis functionality.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any


def analyze_numeric_columns(
    df: pd.DataFrame,
    columns: List[str],
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze numeric columns in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.
    columns : List[str]
        The list of numeric column names to analyze.
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
        
        # Get column data without NaN values for statistics
        column_data = df[column].dropna()
        
        # Basic statistics
        stats = {
            'mean': column_data.mean() if len(column_data) > 0 else None,
            'median': column_data.median() if len(column_data) > 0 else None,
            'std': column_data.std() if len(column_data) > 0 else None,
            'min': column_data.min() if len(column_data) > 0 else None,
            'max': column_data.max() if len(column_data) > 0 else None,
            'range': column_data.max() - column_data.min() if len(column_data) > 0 else None,
            'q1': column_data.quantile(0.25) if len(column_data) > 0 else None,
            'q3': column_data.quantile(0.75) if len(column_data) > 0 else None,
            'iqr': column_data.quantile(0.75) - column_data.quantile(0.25) if len(column_data) > 0 else None,
            'skew': column_data.skew() if len(column_data) > 0 else None,
            'kurtosis': column_data.kurtosis() if len(column_data) > 0 else None,
            'count': len(column_data),
            'missing_count': df[column].isna().sum(),
            'missing_percentage': df[column].isna().mean() * 100,
        }
        
        # Detect outliers using IQR method
        if len(column_data) > 0:
            q1 = stats['q1']
            q3 = stats['q3']
            iqr = stats['iqr']
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = column_data[(column_data < lower_bound) | (column_data > upper_bound)]
            stats['outlier_count'] = len(outliers)
            stats['outlier_percentage'] = len(outliers) / len(column_data) * 100 if len(column_data) > 0 else 0
            stats['outlier_lower_bound'] = lower_bound
            stats['outlier_upper_bound'] = upper_bound
        
        results[column] = stats
    
    return results