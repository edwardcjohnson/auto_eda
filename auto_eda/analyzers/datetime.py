"""
Datetime data analysis functionality.

This module provides functions to analyze datetime columns in a DataFrame.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


def analyze_datetime_columns(
    df: pd.DataFrame,
    columns: List[str],
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze datetime columns in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.
    columns : List[str]
        The list of datetime column names to analyze.
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
        column_data = df[column].dropna()
        
        # Basic statistics
        stats = {
            'count': len(column_data),
            'missing_count': df[column].isna().sum(),
            'missing_percentage': df[column].isna().mean() * 100,
        }
        
        if len(column_data) > 0:
            # Ensure column is datetime type
            if not pd.api.types.is_datetime64_dtype(column_data):
                try:
                    column_data = pd.to_datetime(column_data)
                except Exception:
                    # If conversion fails, skip detailed analysis
                    results[column] = stats
                    continue
            
            # Time range
            stats.update({
                'min_date': column_data.min(),
                'max_date': column_data.max(),
                'range_days': (column_data.max() - column_data.min()).days,
            })
            
            # Distribution by year, month, day of week
            if len(column_data) > 0:
                year_counts = column_data.dt.year.value_counts().sort_index()
                month_counts = column_data.dt.month.value_counts().sort_index()
                weekday_counts = column_data.dt.dayofweek.value_counts().sort_index()
                
                # Convert month numbers to month names
                month_names = {
                    1: 'January', 2: 'February', 3: 'March', 4: 'April',
                    5: 'May', 6: 'June', 7: 'July', 8: 'August',
                    9: 'September', 10: 'October', 11: 'November', 12: 'December'
                }
                month_counts.index = month_counts.index.map(lambda x: month_names.get(x, x))
                
                # Convert day of week numbers to day names
                day_names = {
                    0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
                    4: 'Friday', 5: 'Saturday', 6: 'Sunday'
                }
                weekday_counts.index = weekday_counts.index.map(lambda x: day_names.get(x, x))
                
                stats.update({
                    'year_distribution': year_counts.to_dict(),
                    'month_distribution': month_counts.to_dict(),
                    'weekday_distribution': weekday_counts.to_dict(),
                })
                
                # Hour distribution (if time information is available)
                if any(column_data.dt.time != pd.Timestamp('00:00:00').time()):
                    hour_counts = column_data.dt.hour.value_counts().sort_index()
                    stats['hour_distribution'] = hour_counts.to_dict()
        
        results[column] = stats
    
    return results


def detect_datetime_patterns(
    df: pd.DataFrame,
    datetime_columns: List[str],
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Detect patterns in datetime columns such as seasonality and periodicity.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.
    datetime_columns : List[str]
        The list of datetime column names to analyze.
    **kwargs
        Additional arguments for pattern detection configuration.
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        A dictionary mapping column names to their pattern detection results.
    """
    results = {}
    
    for column in datetime_columns:
        if column not in df.columns:
            continue
        
        # Get column data without NaN values
        column_data = df[column].dropna()
        
        if len(column_data) == 0:
            results[column] = {
                'error': 'No valid datetime data found in column.'
            }
            continue
        
        # Ensure column is datetime type
        if not pd.api.types.is_datetime64_dtype(column_data):
            try:
                column_data = pd.to_datetime(column_data)
            except Exception:
                results[column] = {
                    'error': 'Failed to convert column to datetime type.'
                }
                continue
        
        # Initialize results dictionary
        patterns = {}
        
        # Check for weekly patterns
        weekday_counts = column_data.dt.dayofweek.value_counts().sort_index()
        weekday_percentages = (weekday_counts / len(column_data) * 100).round(2)
        
        # Convert day of week numbers to day names
        day_names = {
            0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
            4: 'Friday', 5: 'Saturday', 6: 'Sunday'
        }
        weekday_percentages.index = weekday_percentages.index.map(lambda x: day_names.get(x, x))
        
        # Check if there's a significant weekly pattern
        max_day_percentage = weekday_percentages.max()
        min_day_percentage = weekday_percentages.min()
        weekly_pattern_strength = max_day_percentage - min_day_percentage
        
        patterns['weekly'] = {
            'distribution': weekday_percentages.to_dict(),
            'strength': weekly_pattern_strength,
            'significant': weekly_pattern_strength > 10,  # Arbitrary threshold
            'most_common_day': weekday_percentages.idxmax(),
            'most_common_day_percentage': max_day_percentage
        }
        
        # Check for monthly patterns
        month_counts = column_data.dt.month.value_counts().sort_index()
        month_percentages = (month_counts / len(column_data) * 100).round(2)
        
        # Convert month numbers to month names
        month_names = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April',
            5: 'May', 6: 'June', 7: 'July', 8: 'August',
            9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }
        month_percentages.index = month_percentages.index.map(lambda x: month_names.get(x, x))
        
        # Check if there's a significant monthly pattern
        max_month_percentage = month_percentages.max()
        min_month_percentage = month_percentages.min()
        monthly_pattern_strength = max_month_percentage - min_month_percentage
        
        patterns['monthly'] = {
            'distribution': month_percentages.to_dict(),
            'strength': monthly_pattern_strength,
            'significant': monthly_pattern_strength > 15,  # Arbitrary threshold
            'most_common_month': month_percentages.idxmax(),
            'most_common_month_percentage': max_month_percentage
        }
        
        # Check for yearly patterns (if data spans multiple years)
        year_counts = column_data.dt.year.value_counts().sort_index()
        if len(year_counts) > 1:
            # Check for month patterns within years
            yearly_patterns = {}
            for year in year_counts.index:
                year_data = column_data[column_data.dt.year == year]
                year_month_counts = year_data.dt.month.value_counts().sort_index()
                year_month_percentages = (year_month_counts / len(year_data) * 100).round(2)
                year_month_percentages.index = year_month_percentages.index.map(lambda x: month_names.get(x, x))
                yearly_patterns[str(year)] = year_month_percentages.to_dict()
            
            patterns['yearly'] = {
                'year_distribution': year_counts.to_dict(),
                'monthly_patterns_by_year': yearly_patterns
            }
        
        # Check for daily patterns (if time information is available)
        if any(column_data.dt.time != pd.Timestamp('00:00:00').time()):
            hour_counts = column_data.dt.hour.value_counts().sort_index()
            hour_percentages = (hour_counts / len(column_data) * 100).round(2)
            
            # Check if there's a significant hourly pattern
            max_hour_percentage = hour_percentages.max()
            min_hour_percentage = hour_percentages.min()
            hourly_pattern_strength = max_hour_percentage - min_hour_percentage
            
            patterns['hourly'] = {
                'distribution': hour_percentages.to_dict(),
                'strength': hourly_pattern_strength,
                'significant': hourly_pattern_strength > 20,  # Arbitrary threshold
                'most_common_hour': int(hour_percentages.idxmax()),
                'most_common_hour_percentage': max_hour_percentage
            }
        
        results[column] = patterns
    
    return results


def analyze_datetime_intervals(
    df: pd.DataFrame,
    datetime_column: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze intervals between consecutive datetime values.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.
    datetime_column : str
        The datetime column name to analyze.
    **kwargs
        Additional arguments for interval analysis configuration.
        
    Returns
    -------
    Dict[str, Any]
        A dictionary with interval analysis results.
    """
    if datetime_column not in df.columns:
        return {'error': f'Column {datetime_column} not found in DataFrame.'}
    
    # Get column data without NaN values
    column_data = df[datetime_column].dropna()
    
    if len(column_data) <= 1:
        return {'error': 'Not enough data points for interval analysis.'}
    
    # Ensure column is datetime type
    if not pd.api.types.is_datetime64_dtype(column_data):
        try:
            column_data = pd.to_datetime(column_data)
        except Exception:
            return {'error': 'Failed to convert column to datetime type.'}
    
    # Sort data by datetime
    sorted_data = column_data.sort_values()
    
    # Calculate intervals
    intervals = sorted_data.diff()[1:]  # Skip first value which will be NaT