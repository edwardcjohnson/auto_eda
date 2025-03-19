"""
Logging utilities for AutoEDA.
"""

import logging
from typing import Optional


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Set up a logger with the specified name and level.
    
    Parameters
    ----------
    name : str
        The name of the logger.
    level : str, optional
        The logging level, by default "INFO".
        
    Returns
    -------
    logging.Logger
        The configured logger.
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    
    # Create console handler if no handlers exist
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(numeric_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
    
    return logger