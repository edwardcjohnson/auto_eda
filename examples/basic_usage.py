#!/usr/bin/env python3
"""
Basic usage example for AutoEDA.

This script creates a synthetic DataFrame with 20 rows and 8 columns of various data types,
initializes AutoEDA with it, runs analysis, visualizes (creating a dummy file), and generates
both HTML and Markdown reports.
"""

import pandas as pd
from auto_eda import AutoEDA

def main():
    # Create a sample DataFrame with mixed types
    data = {
        "numeric1": list(range(20)),
        "numeric2": [x * 0.5 for x in range(20)],
        "categorical1": ["A", "B", "C", "A", "B"] * 4,
        "categorical2": ["cat", "dog", "bird", "fish", "lizard"] * 4,
        "date1": pd.date_range(start="2021-01-01", periods=20, freq="D"),
        "boolean": [True, False] * 10,
        "text": ["This is a sample sentence for testing purposes." for _ in range(20)],
        "id": [100 + x for x in range(20)]
    }
    df = pd.DataFrame(data)
    
    # Initialize AutoEDA with the DataFrame
    eda = AutoEDA(df, report_dir="example_output")
    
    # Run analysis
    analysis_results = eda.analyze()
    
    # Generate visualizations (creates a dummy file so that the output folder is not empty)
    viz_results = eda.visualize(output_dir="example_output")
    
    # Generate reports
    html_report_path = eda.generate_report(output_dir="example_output", report_format="html")
    md_report_path = eda.generate_report(output_dir="example_output", report_format="markdown")
    
    print("Basic analysis complete.")
    print("HTML report generated at:", html_report_path)
    print("Markdown report generated at:", md_report_path)

if __name__ == "__main__":
    main()