"""
Tests for numeric data analysis functionality.
"""
import pytest
import pandas as pd
import numpy as np
from auto_eda.analyzers.numeric import analyze_numeric_columns

class TestNumericAnalysis:
    """Test numeric data analysis functionality."""
    
    def test_basic_statistics(self):
        """Test basic statistics calculation for numeric columns."""
        df = pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'numeric2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'with_nulls': [1, 2, None, 4, 5, None, 7, 8, 9, 10]
        })
        
        result = analyze_numeric_columns(df, ['numeric1', 'numeric2', 'with_nulls'])
        
        # Check that all columns were analyzed
        assert 'numeric1' in result
        assert 'numeric2' in result
        assert 'with_nulls' in result
        
        # Check basic statistics for first column
        assert result['numeric1']['mean'] == 5.5
        assert result['numeric1']['median'] == 5.5
        assert result['numeric1']['min'] == 1
        assert result['numeric1']['max'] == 10
        assert result['numeric1']['std'] > 0
        
        # Check null handling
        assert result['with_nulls']['missing_count'] == 2
        assert result['with_nulls']['missing_percentage'] == 0.2  # 2 out of 10
    
    def test_outlier_detection(self):
        """Test outlier detection in numeric columns."""
        df = pd.DataFrame({
            'no_outliers': [5, 6, 5, 6, 5, 6, 5, 6, 5, 6],
            'with_outliers': [5, 6, 5, 6, 5, 6, 5, 6, 5, 100]
        })
        