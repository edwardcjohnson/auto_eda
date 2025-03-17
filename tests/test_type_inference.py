"""
Tests for the type inference functionality.
"""
import pytest
import pandas as pd
import numpy as np
from auto_eda.type_inference import infer_column_types, DataTypeInference

class TestTypeInference:
    """Test the type inference functionality."""
    
    def test_numeric_inference(self):
        """Test numeric column inference."""
        df = pd.DataFrame({
            'integer': [1, 2, 3, 4, 5],
            'float': [1.1, 2.2, 3.3, 4.4, 5.5],
            'numeric_as_string': ['1', '2', '3', '4', '5'],
            'mixed_numeric': ['1', '2', 'three', '4', '5']
        })
        
        inference_config = {
            "categorical_threshold": 0.1,
            "numeric_detection_strictness": "medium",
            "date_inference": True,
            "id_detection": True
        }
        
        result = infer_column_types(df, inference_config)
        
        # Check integer column
        assert result['integer'].inferred_type == 'numeric'
        assert result['integer'].confidence > 0.8
        
        # Check float column
        assert result['float'].inferred_type == 'numeric'
        assert result['float'].confidence > 0.8
        
        # Check numeric as string column with high strictness
        inference_config["numeric_detection_strictness"] = "high"
        result_high = infer_column_types(df, inference_config)
        assert result_high['numeric_as_string'].inferred_type == 'numeric'
        
        # Check mixed numeric column with high strictness
        assert result_high['mixed_numeric'].inferred_type != 'numeric'
    
    def test_categorical_inference(self):
        """Test categorical column inference."""
        df = pd.DataFrame({
            'low_cardinality': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
            'high_cardinality': [f"val_{i}" for i in range(10)],
            'category_name': ['type_A', 'type_B', 'type_C', 'type_A', 'type_B']
        })
        
        inference_config = {
            "categorical_threshold": 0.3,  # 30% unique values threshold
            "numeric_detection_strictness": "medium",
            "date_inference": True,
            "id_detection": True
        }
        
        result = infer_column_types(df, inference_config)
        
        # Low cardinality should be categorical
        assert result['low_cardinality'].inferred_type == 'categorical'
        
        # High cardinality should not be categorical
        assert result['high_cardinality'].inferred_type != 'categorical'
        
        # Column with 'type' in name should be categorical
        assert result['category_name'].inferred_type == 'categorical'
    
    def test_date_inference(self):
        """Test date column inference."""
        df = pd.DataFrame({
            'iso_date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'us_date': ['01/01/2023', '01/02/2023', '01/03/2023'],
            'date_column': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'not_a_date': ['apple', 'banana', 'cherry']
        })
        
        inference_config = {
            "categorical_threshold": 0.1,
            "numeric_detection_strictness": "medium",
            "date_inference": True,
            "id_detection": True
        }
        
        result = infer_column_types(df, inference_config)
        
        # ISO format dates should be detected
        assert result['iso_date'].inferred_type == 'datetime'
        
        # US format dates should be detected
        assert result['us_date'].inferred_type == 'datetime'
        
        # Column with 'date' in name should be datetime
        assert result['date_column'].inferred_type == 'datetime'
        
        # Non-date column should not be datetime
        assert result['not_a_date'].inferred_type != 'datetime'
        
        # Test with date inference disabled
        inference_config["date_inference"] = False
        result_no_date = infer_column_types(df, inference_config)
        assert result_no_date['iso_date'].inferred_type != 'datetime'