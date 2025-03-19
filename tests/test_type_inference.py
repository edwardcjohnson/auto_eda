# tests/test_type_inference.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from auto_eda.utils.type_inference import infer_column_types

class TestTypeInference:
    def setup_method(self):
        # Create a test DataFrame with various column types
        self.df = pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5],
            'numeric2': [10.5, 20.5, 30.5, 40.5, 50.5],
            'categorical1': ['A', 'B', 'C', 'A', 'B'],
            'categorical2': ['cat', 'dog', 'cat', 'bird', 'dog'],
            'date1': pd.date_range(start='2021-01-01', periods=5),
            'date2': ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', '2021-01-05'],
            'boolean': [True, False, True, True, False],
            'mixed': [1, 'text', 3.14, True, None]
        })
        
    def test_numeric_inference(self):
        # Test numeric column inference
        column_types = infer_column_types(self.df)
        
        # Check that numeric columns are correctly identified
        assert 'numeric1' in column_types['numeric']
        assert 'numeric2' in column_types['numeric']
        
        # Check that non-numeric columns are not in numeric category
        assert 'categorical1' not in column_types['numeric']
        assert 'date1' not in column_types['numeric']
        
    def test_categorical_inference(self):
        # Test categorical column inference
        column_types = infer_column_types(self.df)
        
        # Check that categorical columns are correctly identified
        assert 'categorical1' in column_types['categorical']
        assert 'categorical2' in column_types['categorical']
        assert 'boolean' in column_types['categorical']  # Boolean can be treated as categorical
        
        # Check that non-categorical columns are not in categorical category
        assert 'numeric1' not in column_types['categorical']
        assert 'date1' not in column_types['categorical']
        
    def test_date_inference(self):
        # Test date column inference
        column_types = infer_column_types(self.df)
        
        # Check that date columns are correctly identified
        assert 'date1' in column_types['datetime']
        
        # The string date column might be detected as datetime or as categorical
        # depending on the implementation
        if 'date2' in column_types['datetime']:
            assert 'date2' in column_types['datetime']
        else:
            assert 'date2' in column_types['categorical']
        
        # Check that non-date columns are not in date category
        assert 'numeric1' not in column_types['datetime']
        assert 'categorical1' not in column_types['datetime']