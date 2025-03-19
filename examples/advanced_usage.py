#!/usr/bin/env python3
"""
Advanced usage example for AutoEDA.

This script creates another synthetic DataFrame with 20 rows and 8 columns using different data,
initializes AutoEDA with a custom report directory and log level, and demonstrates potential type overrides.
It then runs analysis, visualization, and finally generates an HTML report.
"""

import pandas as pd
from auto_eda import AutoEDA

def main():
    # Create a sample DataFrame with mixed types and custom data
    data = {
        "numeric1": [x + 0.123 for x in range(20)],
        "numeric2": [x ** 2 for x in range(20)],
        "categorical1": ["X", "Y", "Z", "X", "Y"] * 4,
        "categorical2": ["red", "blue", "green", "red", "blue"] * 4,
        "date1": pd.date_range(start="2022-05-01", periods=20, freq="W"),
        "boolean": [x % 2 == 0 for x in range(20)],
        "text": ["Lorem ipsum dolor sit amet, consectetur adipiscing elit." for _ in range(20)],
        "id": [200 + x for x in range(20)]
    }
    df = pd.DataFrame(data)
    
    # Initialize AutoEDA with custom report directory and debug logging (advanced usage)
    eda = AutoEDA(df, report_dir="advanced_output", log_level="DEBUG")
    
    # (Optional) Override types manually if needed, e.g., treat 'id' as categorical.
    # Here we simply print a message (assuming an override_types method exists).
    if "id" in df.columns:
        print("Advanced usage: 'id' column can be overridden to categorical manually if desired.")
    
    # Run analysis and visualizations
    analysis_results = eda.analyze()
    viz_results = eda.visualize(output_dir="advanced_output")
    
    # Generate an HTML report
    html_report_path = eda.generate_report(output_dir="advanced_output", report_format="html")
    
    print("Advanced analysis complete.")
    print("Advanced HTML report generated at:", html_report_path)

if __name__ == "__main__":
    main()