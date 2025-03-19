"""
Correlation analysis functionality.

This module provides functions to compute correlations between columns in a DataFrame.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


def compute_correlations(
    df: pd.DataFrame,
    numeric_columns: Optional[List[str]] = None,
    method: str = 'pearson',
    threshold: float = 0.0,
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute correlations between numeric columns in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.
    numeric_columns : List[str], optional
        The list of numeric column names to include in correlation analysis.
        If None, all numeric columns will be used.
    method : str, optional
        The correlation method to use ('pearson', 'spearman', or 'kendall'), by default 'pearson'.
    threshold : float, optional
        The minimum absolute correlation value to include in the results, by default 0.0.
    **kwargs
        Additional arguments for analysis configuration.
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        A tuple containing:
        - The correlation matrix as a DataFrame
        - A dictionary with additional correlation analysis results
    """
    # If no numeric columns are specified, use all numeric columns
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    else:
        # Filter out columns that don't exist or aren't numeric
        numeric_columns = [col for col in numeric_columns if col in df.columns]
        numeric_df = df[numeric_columns]
        numeric_columns = numeric_df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_columns:
        return pd.DataFrame(), {'strong_correlations': []}
    
    # Compute correlation matrix
    corr_matrix = df[numeric_columns].corr(method=method)
    
    # Find strong correlations
    strong_correlations = []
    
    # Get upper triangle of correlation matrix (excluding diagonal)
    for i in range(len(numeric_columns)):
        for j in range(i + 1, len(numeric_columns)):
            col1 = numeric_columns[i]
            col2 = numeric_columns[j]
            corr_value = corr_matrix.loc[col1, col2]
            
            # Check if correlation is above threshold
            if abs(corr_value) >= threshold:
                strong_correlations.append({
                    'column1': col1,
                    'column2': col2,
                    'correlation': corr_value,
                    'abs_correlation': abs(corr_value)
                })
    
    # Sort strong correlations by absolute correlation value (descending)
    strong_correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
    
    # Additional correlation analysis
    additional_results = {
        'strong_correlations': strong_correlations,
        'method': method,
        'threshold': threshold,
        'num_strong_correlations': len(strong_correlations)
    }
    
    return corr_matrix, additional_results


def compute_categorical_correlations(
    df: pd.DataFrame,
    categorical_columns: List[str],
    **kwargs
) -> Dict[str, Dict[str, float]]:
    """
    Compute correlations between categorical columns using Cramer's V.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.
    categorical_columns : List[str]
        The list of categorical column names to include in correlation analysis.
    **kwargs
        Additional arguments for analysis configuration.
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        A dictionary mapping column pairs to their Cramer's V correlation values.
    """
    # Filter out columns that don't exist
    categorical_columns = [col for col in categorical_columns if col in df.columns]
    
    if len(categorical_columns) <= 1:
        return {}
    
    # Initialize results dictionary
    results = {}
    
    # Compute Cramer's V for each pair of categorical columns
    for i in range(len(categorical_columns)):
        col1 = categorical_columns[i]
        results[col1] = {}
        
        for j in range(len(categorical_columns)):
            col2 = categorical_columns[j]
            
            # Skip self-correlation
            if i == j:
                results[col1][col2] = 1.0
                continue
            
            # Compute Cramer's V
            try:
                cramers_v = calculate_cramers_v(df[col1], df[col2])
                results[col1][col2] = cramers_v
            except Exception:
                results[col1][col2] = None
    
    return results


def calculate_cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Calculate Cramer's V statistic for categorical correlation.
    
    Parameters
    ----------
    x : pd.Series
        First categorical variable.
    y : pd.Series
        Second categorical variable.
        
    Returns
    -------
    float
        Cramer's V correlation value.
    """
    try:
        from scipy.stats import chi2_contingency
    except ImportError:
        # Return None if scipy is not available
        return None
    
    # Create contingency table
    contingency_table = pd.crosstab(x, y)
    
    # Calculate Chi-square statistic
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    # Calculate Cramer's V
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    
    if min(kcorr-1, rcorr-1) == 0:
        return 0
    
    cramers_v = np.sqrt(phi2corr / min(kcorr-1, rcorr-1))
    return cramers_v


def compute_correlation_with_target(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[List[str]] = None,
    method: str = 'pearson',
    **kwargs
) -> Dict[str, float]:
    """
    Compute correlations between feature columns and a target column.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.
    target_column : str
        The target column name.
    feature_columns : List[str], optional
        The list of feature column names to include in correlation analysis.
        If None, all columns except the target will be used.
    method : str, optional
        The correlation method to use ('pearson', 'spearman', or 'kendall'), by default 'pearson'.
    **kwargs
        Additional arguments for analysis configuration.
        
    Returns
    -------
    Dict[str, float]
        A dictionary mapping feature column names to their correlation with the target.
    """
    if target_column not in df.columns:
        return {}
    
    # If no feature columns are specified, use all columns except the target
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]
    else:
        # Filter out columns that don't exist or are the target
        feature_columns = [col for col in feature_columns if col in df.columns and col != target_column]
    
    if not feature_columns:
        return {}
    
    # Check if target is numeric
    target_is_numeric = pd.api.types.is_numeric_dtype(df[target_column])
    
    results = {}
    
    for col in feature_columns:
        # Check if feature is numeric
        feature_is_numeric = pd.api.types.is_numeric_dtype(df[col])
        
        # If both are numeric, use specified correlation method
        if target_is_numeric and feature_is_numeric:
            try:
                corr = df[[target_column, col]].corr(method=method).loc[target_column, col]
                results[col] = corr
            except Exception:
                results[col] = None
        
        # If target is categorical and feature is numeric, use point-biserial correlation
        elif not target_is_numeric and feature_is_numeric:
            try:
                # Convert categorical target to dummy variables
                dummies = pd.get_dummies(df[target_column], drop_first=False)
                
                # Calculate correlation for each category
                category_corrs = {}
                for category in dummies.columns:
                    corr = df[col].corr(dummies[category], method=method)
                    category_corrs[category] = corr
                
                # Use the maximum absolute correlation
                max_category = max(category_corrs.items(), key=lambda x: abs(x[1]))
                results[col] = max_category[1]
            except Exception:
                results[col] = None
        
        # If both are categorical, use Cramer's V
        elif not target_is_numeric and not feature_is_numeric:
            try:
                cramers_v = calculate_cramers_v(df[target_column], df[col])
                results[col] = cramers_v
            except Exception:
                results[col] = None
        
        # If target is numeric and feature is categorical, use ANOVA-based correlation
        elif target_is_numeric and not feature_is_numeric:
            try:
                # Calculate eta-squared (effect size)
                eta_squared = calculate_eta_squared(df[target_column], df[col])
                results[col] = eta_squared
            except Exception:
                results[col] = None
    
    return results


def calculate_eta_squared(y: pd.Series, x: pd.Series) -> float:
    """
    Calculate eta-squared (effect size) for ANOVA.
    
    Parameters
    ----------
    y : pd.Series
        Numeric variable.
    x : pd.Series
        Categorical variable.
        
    Returns
    -------
    float
        Eta-squared value.
    """
    try:
        from scipy import stats
    except ImportError:
        # Return None if scipy is not available
        return None
    
    # Get unique categories
    categories = x.dropna().unique()
    
    # If there's only one category, return 0
    if len(categories) <= 1:
        return 0
    
    # Group data by category
    groups = [y[x == category].dropna() for category in categories]
    groups = [group for group in groups if len(group) > 0]
    
    # If there are fewer than 2 groups with data, return 0
    if len(groups) < 2:
        return 0
    
    # Perform one-way ANOVA
    f_val, p_val = stats.f_oneway(*groups)
    
    # Calculate eta-squared
    # Sum of squares between groups
    ss_between = sum(len(group) * ((group.mean() - y.mean()) ** 2) for group in groups)
    # Total sum of squares
    ss_total = sum((y - y.mean()) ** 2)
    
    # Eta-squared is the ratio of between-group variance to total variance
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    return eta_squared