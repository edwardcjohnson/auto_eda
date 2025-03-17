"""
Shared fixtures for AutoEDA tests.
"""
import os
import pandas as pd
import pytest
import numpy as np
from pathlib import Path

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'numeric_col': [1, 2, 3, 4, 5, np.nan, 7, 8, 9, 10],
        'categorical_col': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
        'text_col': [
            'This is a sample text.',
            'Another example of text data.',
            'Text analysis is important.',
            'AutoEDA handles text well.',
            'Natural language processing.',
            'Text mining capabilities.',
            'Sentiment analysis included.',
            'Entity recognition works well.',
            'Word clouds are generated.',
            'Text visualization is supported.'
        ],
        'date_col': pd.date_range(start='2023-01-01', periods=10),
        'id_col': ['ID_001', 'ID_002', 'ID_003', 'ID_004', 'ID_005', 
                  'ID_006', 'ID_007', 'ID_008', 'ID_009', 'ID_010'],
        'boolean_col': [True, False, True, True, False, True, False, True, False, True],
        'mixed_col': ['1', '2', 'three', '4', 'five', '6', 'seven', '8', '9', '10']
    })

@pytest.fixture
def sample_csv_path(sample_dataframe, tmp_path):
    """Create a temporary CSV file with sample data."""
    file_path = tmp_path / "sample_data.csv"
    sample_dataframe.to_csv(file_path, index=False)
    return file_path

@pytest.fixture
def default_config():
    """Return a default configuration dictionary."""
    return {
        "sampling": {
            "enabled": True,
            "max_rows": 1000,
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
                "min_df": 1,
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

@pytest.fixture
def config_file_path(default_config, tmp_path):
    """Create a temporary YAML config file."""
    import yaml
    file_path = tmp_path / "config.yaml"
    with open(file_path, 'w') as f:
        yaml.dump(default_config, f)
    return file_path

@pytest.fixture
def report_dir(tmp_path):
    """Create a temporary directory for reports."""
    report_path = tmp_path / "reports"
    report_path.mkdir()
    return report_path