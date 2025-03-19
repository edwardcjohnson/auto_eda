"""
Utility functions for report generation.

This module provides utility functions used by the reporting modules.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import base64
import io
from datetime import datetime


def format_number(value: Any) -> str:
    """
    Format a number for display in reports.
    
    Parameters
    ----------
    value : Any
        The value to format
        
    Returns
    -------
    str
        Formatted string representation of the value
    """
    if pd.isna(value):
        return "N/A"
    
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    
    if isinstance(value, (float, np.floating)):
        if abs(value) < 0.001 or abs(value) >= 10000:
            return f"{value:.4e}"
        else:
            return f"{value:.4f}"
    
    return str(value)


def get_file_extension(file_path: str) -> str:
    """
    Get the file extension from a file path.
    
    Parameters
    ----------
    file_path : str
        The file path
        
    Returns
    -------
    str
        The file extension (without the dot)
    """
    return os.path.splitext(file_path)[1][1:].lower()


def is_image_file(file_path: str) -> bool:
    """
    Check if a file is an image file based on its extension.
    
    Parameters
    ----------
    file_path : str
        The file path
        
    Returns
    -------
    bool
        True if the file is an image file, False otherwise
    """
    image_extensions = ["png", "jpg", "jpeg", "gif", "bmp", "svg"]
    return get_file_extension(file_path) in image_extensions


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64.
    
    Parameters
    ----------
    image_path : str
        Path to the image file
        
    Returns
    -------
    str
        Base64-encoded image data
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def truncate_string(s: str, max_length: int = 50) -> str:
    """
    Truncate a string to a maximum length.
    
    Parameters
    ----------
    s : str
        The string to truncate
    max_length : int, optional
        Maximum length of the truncated string, by default 50
        
    Returns
    -------
    str
        Truncated string
    """
    if len(s) <= max_length:
        return s
    
    return s[:max_length - 3] + "..."