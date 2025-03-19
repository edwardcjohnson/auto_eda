"""
Configuration handling utilities for AutoEDA.

This module provides functions to load and save configuration from various sources
including YAML, JSON, and Python dictionaries.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path


def load_configuration(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load configuration from a file or return default configuration.
    
    Parameters
    ----------
    config_path : str or Path, optional
        Path to the configuration file (YAML or JSON), by default None
        
    Returns
    -------
    Dict[str, Any]
        The loaded configuration or default configuration if no path is provided.
        
    Raises
    ------
    ValueError
        If the file format is not supported or the file doesn't exist.
    """
    # Default configuration
    default_config = {
        "sampling": {
            "enabled": True,
            "max_rows": 10000,
            "random_state": 42
        },
        "type_inference": {
            "categorical_threshold": 0.1,
            "numeric_detection_strictness": "medium",
            "date_inference": True,
            "id_detection": True
        },
        "analysis": {
            "correlation_method": "pearson",
            "outlier_detection": {
                "enabled": True,
                "method": "iqr",
                "threshold": 1.5
            },
            "text_analysis": {
                "max_features": 100,
                "min_df": 2,
                "ngram_range": [1, 2]
            }
        },
        "visualization": {
            "style": "whitegrid",
            "palette": "viridis",
            "figure_size": [10, 6],
            "dpi": 100,
            "interactive": True
        },
        "reporting": {
            "include_code": True,
            "include_recommendations": True,
            "max_rows_in_report": 20
        }
    }
    
    if config_path is None:
        return default_config
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise ValueError(f"Configuration file not found: {config_path}")
    
    # Load based on file extension
    file_extension = config_path.suffix.lower()
    
    try:
        if file_extension in ['.yaml', '.yml']:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
        elif file_extension == '.json':
            with open(config_path, 'r') as file:
                config = json.load(file)
        else:
            raise ValueError(f"Unsupported configuration file format: {file_extension}")
    except Exception as e:
        raise ValueError(f"Error loading configuration file: {e}")
    
    # Merge with default configuration to ensure all required keys exist
    merged_config = _merge_configs(default_config, config)
    
    return merged_config


def save_configuration(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save configuration to a file.
    
    Parameters
    ----------
    config : Dict[str, Any]
        The configuration to save.
    path : str or Path
        The path where to save the configuration.
        
    Raises
    ------
    ValueError
        If the file format is not supported.
    """
    path = Path(path)
    
    # Create directory if it doesn't exist
    os.makedirs(path.parent, exist_ok=True)
    
    # Save based on file extension
    file_extension = path.suffix.lower()
    
    try:
        if file_extension in ['.yaml', '.yml']:
            with open(path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
        elif file_extension == '.json':
            with open(path, 'w') as file:
                json.dump(config, file, indent=4)
        else:
            raise ValueError(f"Unsupported configuration file format: {file_extension}")
    except Exception as e:
        raise ValueError(f"Error saving configuration file: {e}")


def _merge_configs(default_config: Dict[str, Any], user_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge user configuration with default configuration.
    
    Parameters
    ----------
    default_config : Dict[str, Any]
        The default configuration.
    user_config : Dict[str, Any]
        The user-provided configuration.
        
    Returns
    -------
    Dict[str, Any]
        The merged configuration.
    """
    merged = default_config.copy()
    
    for key, value in user_config.items():
        # If the value is a dictionary and the key exists in the default config
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = _merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def load_configuration_from_dict(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load configuration from a dictionary and merge with default configuration.
    
    Parameters
    ----------
    config_dict : Dict[str, Any]
        The configuration dictionary.
        
    Returns
    -------
    Dict[str, Any]
        The merged configuration.
    """
    default_config = load_configuration()
    return _merge_configs(default_config, config_dict)