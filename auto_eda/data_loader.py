"""
Data loading utilities for AutoEDA.

This module provides functions to load data from various sources including:
- CSV files
- Excel files
- JSON files
- Parquet files
- SQL databases
- Pandas DataFrames
"""

import os
import pandas as pd
from typing import Union, Optional, Dict, Any


def load_data_from_source(
    source: Union[str, pd.DataFrame],
    **kwargs
) -> pd.DataFrame:
    """
    Load data from various sources into a pandas DataFrame.
    
    Parameters
    ----------
    source : str or pd.DataFrame
        The data source. Can be a path to a file (CSV, Excel, JSON, Parquet)
        or a pandas DataFrame.
    **kwargs
        Additional arguments to pass to the underlying pandas read functions.
        
    Returns
    -------
    pd.DataFrame
        The loaded data as a pandas DataFrame.
        
    Raises
    ------
    ValueError
        If the source type is not supported or the file doesn't exist.
    """
    if isinstance(source, pd.DataFrame):
        return source.copy()
    
    if not isinstance(source, str):
        raise ValueError(f"Unsupported source type: {type(source)}. Expected string path or DataFrame.")
    
    if not os.path.exists(source):
        raise ValueError(f"File not found: {source}")
    
    # Determine file type from extension
    file_extension = os.path.splitext(source)[1].lower()
    
    # Load based on file extension
    if file_extension == '.csv':
        return pd.read_csv(source, **kwargs)
    elif file_extension in ['.xls', '.xlsx']:
        return pd.read_excel(source, **kwargs)
    elif file_extension == '.json':
        return pd.read_json(source, **kwargs)
    elif file_extension in ['.parquet', '.pq']:
        return pd.read_parquet(source, **kwargs)
    elif file_extension == '.feather':
        return pd.read_feather(source, **kwargs)
    elif file_extension == '.pickle' or file_extension == '.pkl':
        return pd.read_pickle(source, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")


def load_data_from_sql(
    query: str,
    connection,
    **kwargs
) -> pd.DataFrame:
    """
    Load data from a SQL database using a query.
    
    Parameters
    ----------
    query : str
        SQL query to execute.
    connection : SQLAlchemy connectable or str
        Database connection.
    **kwargs
        Additional arguments to pass to pandas.read_sql.
        
    Returns
    -------
    pd.DataFrame
        The query results as a pandas DataFrame.
    """
    return pd.read_sql(query, connection, **kwargs)


def save_dataframe(
    df: pd.DataFrame,
    path: str,
    **kwargs
) -> None:
    """
    Save a DataFrame to a file.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save.
    path : str
        The path where to save the DataFrame.
    **kwargs
        Additional arguments to pass to the underlying pandas write functions.
    
    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    file_extension = os.path.splitext(path)[1].lower()
    
    # Save based on file extension
    if file_extension == '.csv':
        df.to_csv(path, index=False, **kwargs)
    elif file_extension in ['.xls', '.xlsx']:
        df.to_excel(path, index=False, **kwargs)
    elif file_extension == '.json':
        df.to_json(path, **kwargs)
    elif file_extension in ['.parquet', '.pq']:
        df.to_parquet(path, **kwargs)
    elif file_extension == '.feather':
        df.to_feather(path, **kwargs)
    elif file_extension == '.pickle' or file_extension == '.pkl':
        df.to_pickle(path, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")