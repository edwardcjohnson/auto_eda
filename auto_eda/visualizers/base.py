"""
Base visualization class for AutoEDA.

This module provides the base class for all visualizers.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np

class BaseVisualizer:
    """Base class for all visualizers."""
    
    def __init__(
        self,
        style: str = "whitegrid",
        palette: str = "viridis",
        figure_size: Tuple[int, int] = (10, 6),
        dpi: int = 100,
        **kwargs
    ):
        """
        Initialize the BaseVisualizer.
        
        Parameters
        ----------
        style : str, optional
            The seaborn style to use, by default "whitegrid"
        palette : str, optional
            The color palette to use, by default "viridis"
        figure_size : Tuple[int, int], optional
            The default figure size, by default (10, 6)
        dpi : int, optional
            The figure DPI, by default 100
        **kwargs
            Additional visualization parameters
        """
        self.style = style
        self.palette = palette
        self.figure_size = figure_size
        self.dpi = dpi
        self.kwargs = kwargs
        
        # Set up the visualization style
        sns.set_style(style)
        sns.set_palette(palette)
        plt.rcParams["figure.figsize"] = figure_size
        plt.rcParams["figure.dpi"] = dpi
    
    def create_figure(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
        """
        Create a matplotlib figure and axes.
        
        Parameters
        ----------
        nrows : int, optional
            Number of rows in the subplot grid, by default 1
        ncols : int, optional
            Number of columns in the subplot grid, by default 1
        figsize : Tuple[int, int], optional
            Figure size, by default None (uses self.figure_size)
        **kwargs
            Additional parameters to pass to plt.subplots
        
        Returns
        -------
        Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]
            The figure and axes objects
        """
        if figsize is None:
            figsize = self.figure_size
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)
        return fig, axes
    
    def save_figure(
        self,
        fig: plt.Figure,
        filename: str,
        directory: Optional[Union[str, Path]] = None,
        formats: List[str] = ["png"],
        **kwargs
    ) -> Dict[str, str]:
        """
        Save a figure to disk.
        
        Parameters
        ----------
        fig : plt.Figure
            The figure to save
        filename : str
            The base filename (without extension)
        directory : str or Path, optional
            The directory to save to, by default None (current directory)
        formats : List[str], optional
            The file formats to save as, by default ["png"]
        **kwargs
            Additional parameters to pass to fig.savefig
        
        Returns
        -------
        Dict[str, str]
            Dictionary mapping format to saved file path
        """
        if directory is not None:
            os.makedirs(directory, exist_ok=True)
        
        saved_files = {}
        
        for fmt in formats:
            if directory is not None:
                path = os.path.join(directory, f"{filename}.{fmt}")
            else:
                path = f"{filename}.{fmt}"
            
            fig.savefig(path, format=fmt, dpi=self.dpi, bbox_inches="tight", **kwargs)
            saved_files[fmt] = path
        
        return saved_files
    
    def close_figure(self, fig: plt.Figure):
        """
        Close a figure to free memory.
        
        Parameters
        ----------
        fig : plt.Figure
            The figure to close
        """
        plt.close(fig)