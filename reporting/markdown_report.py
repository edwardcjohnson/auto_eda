"""
Markdown report generation functionality.

This module provides functions to generate Markdown reports from EDA results.
"""

import os
import pandas as pd
import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime


def generate_markdown_report(
    df: pd.DataFrame,
    analysis_results: Dict[str, Any],
    output_dir: Union[str, Path],
    report_name: str = "auto_eda_report",
    include_data_sample: bool = True,
    max_rows_sample: int = 10,
    **kwargs
) -> str:
    """
    Generate a Markdown report from EDA results.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame that was analyzed
    analysis_results : Dict[str, Any]
        The results of the EDA analysis
    output_dir : str or Path
        Directory to save the report
    report_name : str, optional
        Name of the report file (without extension), by default "auto_eda_report"
    include_data_sample : bool, optional
        Whether to include a sample of the data in the report, by default True
    max_rows_sample : int, optional
        Maximum number of rows to include in the data sample, by default 10
    **kwargs
        Additional report configuration options
        
    Returns
    -------
    str
        Path to the generated Markdown report
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Markdown content
    md_content = []
    
    # Add report header
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md_content.append(f"# AutoEDA Report\n")
    md_content.append(f"**Generated:** {current_time}  ")
    md_content.append(f"**Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns\n")
    
    # Add data sample if requested
    if include_data_sample:
        sample_df = df.head(max_rows_sample)
        
        md_content.append(f"## Data Sample\n")
        md_content.append(f"First {min(max_rows_sample, df.shape[0])} rows of the dataset:\n")
        
        # Convert DataFrame to markdown table
        md_content.append(dataframe_to_markdown(sample_df))
        md_content.append("\n")
    
    # Add dataset overview
    if "overview" in analysis_results:
        overview = analysis_results["overview"]
        
        md_content.append("## Dataset Overview\n")
        
        if "column_types" in overview:
            column_types = overview["column_types"]
            md_content.append("### Column Types\n")
            
            md_content.append("| Type | Count | Columns |")
            md_content.append("| ---- | ----- | ------- |")
            
            for col_type, info in column_types.items():
                md_content.append(f"| {col_type} | {info['count']} | {', '.join(info['columns'])} |")
            
            md_content.append("\n")
        
        if "missing_values" in overview:
            missing_values = overview["missing_values"]
            md_content.append("### Missing Values\n")
            
            md_content.append("| Column | Missing Count | Missing Percentage |")
            md_content.append("| ------ | ------------- | ------------------ |")
            
            for col, info in missing_values.items():
                md_content.append(f"| {col} | {info['count']} | {info['percentage']:.2f}% |")
            
            md_content.append("\n")
    
    # Add numeric analysis results
    if "numeric_analysis" in analysis_results:
        numeric_analysis = analysis_results["numeric_analysis"]
        
        md_content.append("## Numeric Columns Analysis\n")
        
        for column, stats in numeric_analysis.items():
            md_content.append(f"### {column}\n")
            
            md_content.append("| Statistic | Value |")
            md_content.append("| --------- | ----- |")
            md_content.append(f"| Count | {stats.get('count', 'N/A')} |")
            md_content.append(f"| Mean | {stats.get('mean', 'N/A')} |")
            md_content.append(f"| Std Dev | {stats.get('std', 'N/A')} |")
            md_content.append(f"| Min | {stats.get('min', 'N/A')} |")
            md_content.append(f"| 25% | {stats.get('25%', 'N/A')} |")
            md_content.append(f"| Median | {stats.get('50%', 'N/A')} |")
            md_content.append(f"| 75% | {stats.get('75%', 'N/A')} |")
            md_content.append(f"| Max | {stats.get('max', 'N/A')} |")
            
            md_content.append("\n")
            
            # Add distribution plot if available
            if "plots" in stats and "distribution" in stats["plots"]:
                plot_path = stats["plots"]["distribution"]
                if os.path.exists(plot_path):
                    # Get relative path for the image
                    rel_path = os.path.relpath(plot_path, output_dir)
                    md_content.append(f"![Distribution of {column}]({rel_path})")
                    md_content.append(f"*Distribution of {column}*\n")
    
    # Add categorical analysis results
    if "categorical_analysis" in analysis_results:
        categorical_analysis = analysis_results["categorical_analysis"]
        
        md_content.append("## Categorical Columns Analysis\n")
        
        for column, stats in categorical_analysis.items():
            md_content.append(f"### {column}\n")
            
            md_content.append(f"**Unique Values:** {stats.get('unique_count', 'N/A')}  ")
            md_content.append(f"**Most Common:** {stats.get('most_common_value', 'N/A')} ({stats.get('most_common_percentage', 'N/A'):.2f}%)\n")
            
            # Add value counts table
            if "value_counts" in stats:
                value_counts = stats["value_counts"]
                md_content.append("#### Value Counts\n")
                
                md_content.append("| Value | Count | Percentage |")
                md_content.append("| ----- | ----- | ---------- |")
                
                for value, count in value_counts.items():
                    percentage = count / stats.get('count', 1) * 100
                    md_content.append(f"| {value} | {count} | {percentage:.2f}% |")
                
                md_content.append("\n")
            
            # Add bar plot if available
            if "plots" in stats and "bar" in stats["plots"]:
                plot_path = stats["plots"]["bar"]
                if os.path.exists(plot_path):
                    # Get relative path for the image
                    rel_path = os.path.relpath(plot_path, output_dir)
                    md_content.append(f"![Distribution of {column}]({rel_path})")
                    md_content.append(f"*Distribution of {column}*\n")
    
    # Add correlation analysis results
    if "correlation_analysis" in analysis_results:
        correlation_analysis = analysis_results["correlation_analysis"]
        
        md_content.append("## Correlation Analysis\n")
        
        # Add correlation matrix plot if available
        if "correlation_matrix_plot" in correlation_analysis:
            plot_path = correlation_analysis["correlation_matrix_plot"]
            if os.path.exists(plot_path):
                # Get relative path for the image
                rel_path = os.path.relpath(plot_path, output_dir)
                md_content.append(f"![Correlation Matrix]({rel_path})")
                md_content.append("*Correlation Matrix*\n")
        
        # Add strong correlations table
        if "strong_correlations" in correlation_analysis:
            strong_correlations = correlation_analysis["strong_correlations"]
            if strong_correlations:
                md_content.append("### Strong Correlations\n")
                
                md_content.append("| Column 1 | Column 2 | Correlation |")
                md_content.append("| -------- | -------- | ----------- |")
                
                for corr in strong_correlations:
                    md_content.append(f"| {corr['column1']} | {corr['column2']} | {corr['correlation']:.4f} |")
                
                md_content.append("\n")
    
    # Add footer
    md_content.append("---\n")
    md_content.append("*Generated by AutoEDA*")
    
    # Combine all Markdown content
    full_md = "\n".join(md_content)
    
    # Write Markdown to file
    report_path = os.path.join(output_dir, f"{report_name}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_md)
    
    print(f"Markdown report saved to: {report_path}")
    
    return report_path


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """
    Convert a pandas DataFrame to a Markdown table.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to convert
        
    Returns
    -------
    str
        Markdown table representation of the DataFrame
    """
    # Create header row
    header = "| " + " | ".join(str(col) for col in df.columns) + " |"
    
    # Create separator row
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    
    # Create data rows
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(val) for val in row) + " |")
    
    # Combine all rows
    return "\n".join([header, separator] + rows)