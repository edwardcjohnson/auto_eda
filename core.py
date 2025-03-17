#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core functionality for the AutoEDA package.

This module contains the main AutoEDA class that orchestrates the entire
exploratory data analysis process.
"""

import gc
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from tqdm.auto import tqdm

from auto_eda.data_loader import load_data_from_source
from auto_eda.type_inference import infer_column_types, DataTypeInference
from auto_eda.utils.config import load_configuration
from auto_eda.utils.logger import setup_logger
from auto_eda.analyzers import (
    analyze_numeric_columns,
    analyze_categorical_columns,
    analyze_text_columns,
    analyze_datetime_columns,
    compute_correlations
)
from auto_eda.visualizers import (
    create_distribution_plots,
    create_relationship_plots,
    create_categorical_plots,
    create_text_visualizations
)
from auto_eda.reporting import generate_html_report


class AutoEDA:
    """
    Automated Exploratory Data Analysis class that provides comprehensive
    analysis capabilities for both structured and unstructured data.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        report_dir: str = "./reports",
        log_level: str = "INFO",
    ) -> None:
        """
        Initialize the AutoEDA instance.

        Args:
            config_path: Path to configuration file (YAML or JSON)
            report_dir: Directory to save reports
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = setup_logger("AutoEDA", log_level)
        
        # Load configuration
        self.config = load_configuration(config_path)
        
        # Initialize data attributes
        self.df = None
        self.df_original = None
        self.df_sample = None
        self.column_stats = {}
        self.inferred_types = {}
        self.text_columns = []
        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        self.boolean_columns = []
        self.id_columns = []
        self.analysis_results = {}
        self.visualizations = []
        
        self.logger.info("AutoEDA initialized successfully")

    def load_data(
        self,
        data_source: Union[str, pd.DataFrame],
        infer_types: bool = True,
        categorical_threshold: Optional[float] = None,
        date_inference: Optional[bool] = None,
        numeric_detection_strictness: Optional[str] = None,
        sample: bool = True,
    ) -> None:
        """
        Load data from various sources and perform initial processing.

        Args:
            data_source: Path to data file or pandas DataFrame
            infer_types: Whether to perform intelligent type inference
            categorical_threshold: Threshold for categorical variable detection
            date_inference: Whether to infer date columns
            numeric_detection_strictness: Strictness level for numeric detection
            sample: Whether to sample large datasets
        """
        self.logger.info(f"Loading data from {data_source if isinstance(data_source, str) else 'DataFrame'}")
        
        # Load data from source
        self.df_original = load_data_from_source(data_source)
        
        # Create a working copy
        self.df = self.df_original.copy()
        
        # Sample large datasets if enabled
        if sample and self.config["sampling"]["enabled"] and len(self.df) > self.config["sampling"]["max_rows"]:
            sample_size = self.config["sampling"]["max_rows"]
            self.df_sample = self.df.sample(
                n=sample_size, 
                random_state=self.config["sampling"]["random_state"]
            )
            self.logger.info(f"Created sample of {sample_size} rows from {len(self.df)} total rows")