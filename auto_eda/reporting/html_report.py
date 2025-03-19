"""
HTML report generation functionality.
This module provides functions to generate HTML reports from EDA results.
"""
import os
import pandas as pd
import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import base64
import io
from datetime import datetime

def format_percentage(val) -> str:
    try:
        return f"{float(val):.2f}%"
    except (ValueError, TypeError):
        return "N/A"

def generate_html_report(
    df: pd.DataFrame,
    analysis_results: Dict[str, Any],
    output_dir: Union[str, Path],
    report_name: str = "auto_eda_report",
    include_data_sample: bool = True,
    max_rows_sample: int = 10,
    **kwargs
) -> str:
    """
    Generate an HTML report from EDA results.
    """
    os.makedirs(output_dir, exist_ok=True)
    html_content = []
    html_content.append("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AutoEDA Report</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }
            .container { max-width: 1200px; margin: 0 auto; }
            h1, h2, h3, h4 { color: #2c3e50; }
            h1 { border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            h2 { border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; margin-top: 30px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            tr:hover { background-color: #f5f5f5; }
            .summary-box { background-color: #f8f9fa; border-left: 4px solid #3498db; padding: 15px; margin: 20px 0; }
            .warning { background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 20px 0; }
            .image-container { margin: 20px 0; text-align: center; }
            img { max-width: 100%; height: auto; border: 1px solid #ddd; }
            .footer { margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center;
                      font-size: 0.9em; color: #7f8c8d; }
            .tab { overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }
            .tab button { background-color: inherit; float: left; border: none; outline: none; cursor: pointer;
                          padding: 14px 16px; transition: 0.3s; }
            .tab button:hover { background-color: #ddd; }
            .tab button.active { background-color: #ccc; }
            .tabcontent { display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; }
        </style>
    </head>
    <body>
        <div class="container">
    """)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_content.append(f"""
            <h1>AutoEDA Report</h1>
            <div class="summary-box">
                <p><strong>Generated:</strong> {current_time}</p>
                <p><strong>Dataset Shape:</strong> {df.shape[0]} rows × {df.shape[1]} columns</p>
            </div>
    """)
    if include_data_sample:
        sample_df = df.head(max_rows_sample)
        sample_html = sample_df.to_html(classes="table table-striped", index=True)
        html_content.append(f"""
            <h2>Data Sample</h2>
            <p>First {min(max_rows_sample, df.shape[0])} rows of the dataset:</p>
            {sample_html}
        """)
    if "overview" in analysis_results:
        overview = analysis_results["overview"]
        html_content.append("""
            <h2>Dataset Overview</h2>
        """)
        if "column_types" in overview:
            column_types = overview["column_types"]
            html_content.append("""
                <h3>Column Types</h3>
                <table>
                    <tr>
                        <th>Type</th>
                        <th>Count</th>
                        <th>Columns</th>
                    </tr>
            """)
            for col_type, info in column_types.items():
                html_content.append(f"""
                    <tr>
                        <td>{col_type}</td>
                        <td>{info['count']}</td>
                        <td>{', '.join(info['columns'])}</td>
                    </tr>
                """)
            html_content.append("</table>")
        if "missing_values" in overview:
            missing_values = overview["missing_values"]
            html_content.append("""
                <h3>Missing Values</h3>
                <table>
                    <tr>
                        <th>Column</th>
                        <th>Missing Count</th>
                        <th>Missing Percentage</th>
                    </tr>
            """)
            for col, info in missing_values.items():
                html_content.append(f"""
                    <tr>
                        <td>{col}</td>
                        <td>{info['count']}</td>
                        <td>{info['percentage']:.2f}%</td>
                    </tr>
                """)
            html_content.append("</table>")
    if "numeric_analysis" in analysis_results:
        numeric_analysis = analysis_results["numeric_analysis"]
        html_content.append("""
            <h2>Numeric Columns Analysis</h2>
        """)
        for column, stats in numeric_analysis.items():
            html_content.append(f"""
                <h3>{column}</h3>
                <div class="summary-box">
                    <table>
                        <tr>
                            <th>Statistic</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Count</td>
                            <td>{stats.get('count', 'N/A')}</td>
                        </tr>
                        <tr>
                            <td>Mean</td>
                            <td>{stats.get('mean', 'N/A')}</td>
                        </tr>
                        <tr>
                            <td>Std Dev</td>
                            <td>{stats.get('std', 'N/A')}</td>
                        </tr>
                        <tr>
                            <td>Min</td>
                            <td>{stats.get('min', 'N/A')}</td>
                        </tr>
                        <tr>
                            <td>25%</td>
                            <td>{stats.get('25%', 'N/A')}</td>
                        </tr>
                        <tr>
                            <td>Median</td>
                            <td>{stats.get('50%', 'N/A')}</td>
                        </tr>
                        <tr>
                            <td>75%</td>
                            <td>{stats.get('75%', 'N/A')}</td>
                        </tr>
                        <tr>
                            <td>Max</td>
                            <td>{stats.get('max', 'N/A')}</td>
                        </tr>
                    </table>
                </div>
            """)
            if "plots" in stats and "distribution" in stats["plots"]:
                plot_path = stats["plots"]["distribution"]
                if os.path.exists(plot_path):
                    with open(plot_path, "rb") as img_file:
                        img_data = base64.b64encode(img_file.read()).decode()
                    html_content.append(f"""
                        <div class="image-container">
                            <img src="data:image/png;base64,{img_data}" alt="Distribution of {column}">
                            <p>Distribution of {column}</p>
                        </div>
                    """)
    if "categorical_analysis" in analysis_results:
        categorical_analysis = analysis_results["categorical_analysis"]
        html_content.append("""
            <h2>Categorical Columns Analysis</h2>
        """)
        for column, stats in categorical_analysis.items():
            html_content.append(f"""
                <h3>{column}</h3>
                <div class="summary-box">
                    <p><strong>Unique Values:</strong> {stats.get('unique_count', 'N/A')}</p>
                    <p><strong>Most Common:</strong> {stats.get('most_common_value', 'N/A')} ({format_percentage(stats.get('most_common_percentage', None))})</p>
                </div>
            """)
            if "value_counts" in stats:
                value_counts = stats["value_counts"]
                html_content.append("""
                    <h4>Value Counts</h4>
                    <table>
                        <tr>
                            <th>Value</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
                """)
                for value, count in value_counts.items():
                    percentage = count / stats.get('count', 1) * 100
                    html_content.append(f"""
                        <tr>
                            <td>{value}</td>
                            <td>{count}</td>
                            <td>{percentage:.2f}%</td>
                        </tr>
                    """)
                html_content.append("</table>")
            if "plots" in stats and "bar" in stats["plots"]:
                plot_path = stats["plots"]["bar"]
                if os.path.exists(plot_path):
                    with open(plot_path, "rb") as img_file:
                        img_data = base64.b64encode(img_file.read()).decode()
                    html_content.append(f"""
                        <div class="image-container">
                            <img src="data:image/png;base64,{img_data}" alt="Distribution of {column}">
                            <p>Distribution of {column}</p>
                        </div>
                    """)
    if "correlation_analysis" in analysis_results:
        correlation_analysis = analysis_results["correlation_analysis"]
        html_content.append("""
            <h2>Correlation Analysis</h2>
        """)
        if "correlation_matrix_plot" in correlation_analysis:
            plot_path = correlation_analysis["correlation_matrix_plot"]
            if os.path.exists(plot_path):
                with open(plot_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                html_content.append(f"""
                    <div class="image-container">
                        <img src="data:image/png;base64,{img_data}" alt="Correlation Matrix">
                        <p>Correlation Matrix</p>
                    </div>
                """)
        if "strong_correlations" in correlation_analysis:
            strong_correlations = correlation_analysis["strong_correlations"]
            if strong_correlations:
                html_content.append("""
                    <h3>Strong Correlations</h3>
                    <table>
                        <tr>
                            <th>Column 1</th>
                            <th>Column 2</th>
                            <th>Correlation</th>
                        </tr>
                """)
                for corr in strong_correlations:
                    html_content.append(f"""
                        <tr>
                            <td>{corr['column1']}</td>
                            <td>{corr['column2']}</td>
                            <td>{corr['correlation']:.4f}</td>
                        </tr>
                    """)
                html_content.append("</table>")
    html_content.append("""
            <div class="footer">
                <p>Generated by AutoEDA</p>
            </div>
        </div>
        <script>
            function openTab(evt, tabName) {
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tabcontent");
                for (i = 0; i < tabcontent.length; i++) {
                    tabcontent[i].style.display = "none";
                }
                tablinks = document.getElementsByClassName("tablinks");
                for (i = 0; i < tablinks.length; i++) {
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }
                document.getElementById(tabName).style.display = "block";
                evt.currentTarget.className += " active";
            }
        </script>
        </body>
        </html>
    """)
    full_html = "\n".join(html_content)
    report_path = os.path.join(output_dir, f"{report_name}.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"HTML report saved to: {report_path}")
    return report_path