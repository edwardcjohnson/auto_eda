from setuptools import setup, find_packages
import os

# Read the contents of README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Read version from version.py
about = {}
with open(os.path.join("auto_eda", "version.py"), encoding="utf-8") as f:
    exec(f.read(), about)

setup(
    name="auto_eda",
    version=about["__version__"],
    description="Automated Exploratory Data Analysis tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Edward Johnson",
    url="https://github.com/edwardcjohnson/auto_eda",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "scipy>=1.5.0",
        "scikit-learn>=0.23.0",
    ],
    extras_require={
        "interactive": ["plotly>=4.9.0"],
        "wordcloud": ["wordcloud>=1.8.0"],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=20.8b1",
            "flake8>=3.8.0",
            "mypy>=0.782",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    keywords="data analysis, visualization, exploratory data analysis, eda, data science",
    project_urls={
        "Bug Reports": "https://github.com/edwardcjohnson/auto_eda/issues",
        "Source": "https://github.com/edwardcjohnson/auto_eda",
        "Documentation": "https://auto-eda.readthedocs.io/",
    },
    entry_points={
        "console_scripts": [
            "auto_eda=auto_eda.cli:main",
        ],
    },
)