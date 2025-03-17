"""
Tests for the core AutoEDA functionality.
"""
import pytest
import pandas as pd
from auto_eda import AutoEDA

class TestAutoEDA:
    """Test the main AutoEDA class."""
    
    def test_initialization(self, config_file_path, report_dir):
        """Test that AutoEDA initializes correctly."""
        eda = AutoEDA(config_path=config_file_path, report_dir=str(report_dir))
        assert eda is not None
        assert eda.config is not None
        assert eda.report_dir == report_dir
    
    def test_load_data_from_dataframe(self, sample_dataframe):
        """Test loading data from a pandas DataFrame."""
        eda = AutoEDA()
        eda.load_data(sample_dataframe, infer_types=False)
        assert eda.df is not None
        assert eda.df.equals(sample_dataframe)
        assert eda.df_original.equals(sample_dataframe)
    
    def test_load_data_from_csv(self, sample_csv_path):
        """Test loading data from a CSV file."""
        eda = AutoEDA()
        eda.load_data(str(sample_csv_path), infer_types=False)
        assert eda.df is not None
        assert len(eda.df) == 10  # Our sample has 10 rows
        assert 'numeric_col' in eda.df.columns
    
    def test_type_inference(self, sample_dataframe):
        """Test data type inference."""
        eda = AutoEDA()
        eda.load_data(sample_dataframe, infer_types=True)
        
        # Check that types were inferred
        assert len(eda.inferred_types) > 0
        
        # Check numeric column detection
        assert 'numeric_col' in eda.numeric_columns
        
        # Check categorical column detection
        assert 'categorical_col' in eda.categorical_columns
        
        # Check date column detection
        assert 'date_col' in eda.datetime_columns
        
        # Check boolean column detection
        assert 'boolean_col' in eda.boolean_columns
    
    def test_analyze_method(self, sample_dataframe):
        """Test the analyze method."""
        eda = AutoEDA()
        eda.load_data(sample_dataframe, infer_types=True)
        eda.analyze()
        
        # Check that analysis results exist
        assert eda.analysis_results is not None
        assert len(eda.analysis_results) > 0
    
    def test_visualize_method(self, sample_dataframe, report_dir):
        """Test the visualize method."""
        eda = AutoEDA(report_dir=str(report_dir))
        eda.load_data(sample_dataframe, infer_types=True)
        eda.analyze()
        eda.visualize()
        
        # Check that visualizations were created
        assert len(eda.visualizations) > 0
    
    def test_generate_report(self, sample_dataframe, report_dir):
        """Test report generation."""
        eda = AutoEDA(report_dir=str(report_dir))
        eda.load_data(sample_dataframe, infer_types=True)
        eda.analyze()
        eda.visualize()
        report_path = eda.generate_report(title="Test Report")
        
        # Check that the report file exists
        assert report_path.exists()
    
    def test_run_analysis_and_report(self, sample_dataframe, report_dir):
        """Test the combined analysis and report method."""
        eda = AutoEDA(report_dir=str(report_dir))
        report_path = eda.run_analysis_and_report(
            data=sample_dataframe,
            title="Combined Test",
            text_columns=["text_col"]
        )
        
        # Check that the report file exists
        assert report_path.exists()