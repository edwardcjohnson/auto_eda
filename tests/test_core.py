# tests/test_core.py
import pytest
import pandas as pd
import numpy as np
import os
from auto_eda.core import AutoEDA

class TestAutoEDA:
    def setup_method(self):
        # Create a test DataFrame
        self.df = pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5],
            'numeric2': [10.5, 20.5, 30.5, 40.5, 50.5],
            'categorical1': ['A', 'B', 'C', 'A', 'B'],
            'categorical2': ['cat', 'dog', 'cat', 'bird', 'dog'],
            'date1': pd.date_range(start='2021-01-01', periods=5),
            'boolean': [True, False, True, True, False]
        })
        
        # Create a temporary directory for output
        self.output_dir = "test_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def teardown_method(self):
        # Clean up temporary files if needed
        pass
        
    def test_type_inference(self):
        # Test that AutoEDA correctly infers column types
        auto_eda = AutoEDA(self.df)
        
        # Check that column types are inferred
        assert hasattr(auto_eda, 'column_types')
        assert isinstance(auto_eda.column_types, dict)
        
        # Check that numeric columns are correctly identified
        assert 'numeric1' in auto_eda.column_types['numeric']
        assert 'numeric2' in auto_eda.column_types['numeric']
        
        # Check that categorical columns are correctly identified
        assert 'categorical1' in auto_eda.column_types['categorical']
        assert 'categorical2' in auto_eda.column_types['categorical']
        
        # Check that date columns are correctly identified
        assert 'date1' in auto_eda.column_types['datetime']
        
    def test_analyze_method(self):
        # Test the analyze method
        auto_eda = AutoEDA(self.df)
        results = auto_eda.analyze()
        
        # Check that results contain expected sections
        assert 'overview' in results
        assert 'numeric_analysis' in results
        assert 'categorical_analysis' in results
        
        # Check that numeric analysis contains expected columns
        assert 'numeric1' in results['numeric_analysis']
        assert 'numeric2' in results['numeric_analysis']
        
        # Check that categorical analysis contains expected columns
        assert 'categorical1' in results['categorical_analysis']
        assert 'categorical2' in results['categorical_analysis']
        
    def test_visualize_method(self):
        # Test the visualize method
        auto_eda = AutoEDA(self.df)
        auto_eda.analyze()  # Need to analyze first
        
        # Visualize with output directory
        viz_results = auto_eda.visualize(output_dir=self.output_dir)
        
        # Check that visualization results contain expected sections
        assert 'numeric_visualizations' in viz_results
        assert 'categorical_visualizations' in viz_results
        
        # Check that files were created
        assert os.path.exists(self.output_dir)
        assert len(os.listdir(self.output_dir)) > 0
        
    def test_generate_report(self):
        # Test the generate_report method
        auto_eda = AutoEDA(self.df)
        auto_eda.analyze()  # Need to analyze first
        auto_eda.visualize(output_dir=self.output_dir)  # Need to visualize first
        
        # Generate HTML report
        html_path = auto_eda.generate_report(
            output_dir=self.output_dir,
            report_format='html'
        )
        
        # Check that report was created
        assert os.path.exists(html_path)
        
        # Generate Markdown report
        md_path = auto_eda.generate_report(
            output_dir=self.output_dir,
            report_format='markdown'
        )
        
        # Check that report was created
        assert os.path.exists(md_path)
        
    def test_run_analysis_and_report(self):
        # Test the run method that does everything in one go
        auto_eda = AutoEDA(self.df)
        
        # Run analysis and generate report
        results = auto_eda.run(
            output_dir=self.output_dir,
            report_format='html'
        )
        
        # Check that results contain expected sections
        assert 'analysis_results' in results
        assert 'visualization_results' in results
        assert 'report_path' in results
        
        # Check that report was created
        assert os.path.exists(results['report_path'])