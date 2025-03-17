# tests/test_analyzers/test_numeric.py
import pytest
import pandas as pd
import numpy as np
from auto_eda.analyzers.numeric import analyze_numeric_columns

class TestNumericAnalysis:
    def setup_method(self):
        # Create a test DataFrame with numeric columns
        self.df = pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5],
            'numeric2': [10.5, 20.5, 30.5, 40.5, 50.5],
            'numeric3': [100, 200, 300, 400, 500],
            'categorical': ['A', 'B', 'C', 'A', 'B']
        })
        
    def test_basic_statistics(self):
        # Analyze numeric columns
        results = analyze_numeric_columns(self.df)  # No need to pass columns explicitly
        
        # Check that all numeric columns are analyzed
        assert set(results.keys()) == {'numeric1', 'numeric2', 'numeric3'}
        
        # Check basic statistics for numeric1
        assert results['numeric1']['count'] == 5
        assert results['numeric1']['mean'] == 3.0
        assert results['numeric1']['std'] == pytest.approx(1.5811, abs=0.001)
        assert results['numeric1']['min'] == 1.0
        assert results['numeric1']['max'] == 5.0
        
        # Check basic statistics for numeric2
        assert results['numeric2']['count'] == 5
        assert results['numeric2']['mean'] == 30.5
        assert results['numeric2']['std'] == pytest.approx(15.811, abs=0.001)
        assert results['numeric2']['min'] == 10.5
        assert results['numeric2']['max'] == 50.5
        
        # Check percentiles - this is likely where the error is
        # Use pytest.approx for floating point comparisons
        assert results['numeric1']['25%'] == pytest.approx(2.0, abs=0.001)
        assert results['numeric1']['50%'] == pytest.approx(3.0, abs=0.001)
        assert results['numeric1']['75%'] == pytest.approx(4.0, abs=0.001)
        
        assert results['numeric2']['25%'] == pytest.approx(20.5, abs=0.001)
        assert results['numeric2']['50%'] == pytest.approx(30.5, abs=0.001)
        assert results['numeric2']['75%'] == pytest.approx(40.5, abs=0.001)