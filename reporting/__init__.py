"""
Reporting modules for generating EDA reports.

This package contains modules for creating various types of reports:
- HTML reports with interactive visualizations
- Markdown reports for documentation
- Utility functions for report generation
"""

from auto_eda.reporting.html_report import generate_html_report
from auto_eda.reporting.markdown_report import generate_markdown_report

__all__ = [
    "generate_html_report",
    "generate_markdown_report",
]