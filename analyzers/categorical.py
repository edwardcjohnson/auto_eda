"""
Categorical data analysis functionality.

This module provides functions to analyze categorical columns in a DataFrame.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


def analyze_categorical_columns(
    df: pd.DataFrame,
    columns: List[str],
    max_categories: int = 20,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze categorical columns in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.
    columns : List[str]
        The list of categorical column names to analyze.
    max_categories : int, optional
        Maximum number of categories to include in the frequency table, by default 20.
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
        
        # Get column data
        column_data = df[column]
        
        # Basic statistics
        value_counts = column_data.value_counts()
        value_counts_normalized = column_data.value_counts(normalize=True)
        
        # Limit the number of categories in the results
        if len(value_counts) > max_categories:
            # Keep the top categories and group the rest as "Other"
            top_categories = value_counts.nlargest(max_categories - 1)
            other_count = value_counts.sum() - top_categories.sum()
            
            top_categories_normalized = value_counts_normalized.nlargest(max_categories - 1)
            other_normalized = value_counts_normalized.sum() - top_categories_normalized.sum()
            
            # Create new Series with "Other" category
            value_counts = pd.concat([top_categories, pd.Series([other_count], index=["Other"])])
            value_counts_normalized = pd.concat([
                top_categories_normalized, 
                pd.Series([other_normalized], index=["Other"])
            ])
        
        # Create frequency table
        frequency_table = pd.DataFrame({
            'count': value_counts,
            'percentage': value_counts_normalized * 100
        })
        
        # Calculate statistics
        stats = {
            'unique_count': column_data.nunique(),
            'missing_count': column_data.isna().sum(),
            'missing_percentage': column_data.isna().mean() * 100,
            'mode': column_data.mode()[0] if not column_data.mode().empty else None,
            'mode_count': value_counts.iloc[0] if not value_counts.empty else 0,
            'mode_percentage': value_counts_normalized.iloc[0] * 100 if not value_counts_normalized.empty else 0,
            'entropy': calculate_entropy(value_counts_normalized) if not value_counts_normalized.empty else 0,
            'frequency_table': frequency_table.reset_index().rename(columns={'index': 'value'}).to_dict('records')
        }
        
        results[column] = stats
    
    return results


def calculate_entropy(probabilities):
    """
    Calculate the entropy of a probability distribution.
    
    Parameters
    ----------
    probabilities : pd.Series
        Series of probabilities.
        
    Returns
    -------
    float
        The entropy value.
    """
    # Filter out zero probabilities to avoid log(0)
    probabilities = probabilities[probabilities > 0]
    return -sum(p * np.log2(p) for p in probabilities)


def analyze_categorical_relationships(
    df: pd.DataFrame,
    target_column: str,
    categorical_columns: List[str],
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze relationships between categorical columns and a target column.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.
    target_column : str
        The target column name.
    categorical_columns : List[str]
        The list of categorical column names to analyze.
    **kwargs
        Additional arguments for analysis configuration.
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        A dictionary mapping column names to their relationship analysis results.
    """
    results = {}
    
    if target_column not in df.columns:
        return results
    
    for column in categorical_columns:
        if column not in df.columns or column == target_column:
            continue
        
        # Create contingency table
        contingency_table = pd.crosstab(
            df[column], 
            df[target_column],
            normalize='index'
        ) * 100
        
        # Calculate chi-square test if target is categorical
        chi2_result = None
        if df[target_column].nunique() <= 20:  # Only for categorical targets
            try:
                from scipy.stats import chi2_contingency
                observed = pd.crosstab(df[column], df[target_column])
                chi2, p, dof, expected = chi2_contingency(observed)
                chi2_result = {
                    'chi2': chi2,
                    'p_value': p,
                    'degrees_of_freedom': dof
                }
            except Exception:
                # If chi-square test fails, just continue without it
                pass
        
        results[column] = {
            'contingency_table': contingency_table.reset_index().to_dict('records'),
            'chi2_test': chi2_result
        }
    
    return results