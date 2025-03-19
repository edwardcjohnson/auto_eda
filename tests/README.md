# AutoEDA Tests

This directory contains the test suite for the AutoEDA package. The tests are organized to verify all aspects of the package's functionality, from data loading to report generation.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                  # Shared fixtures
├── test_core.py                 # Tests for core functionality
├── test_type_inference.py       # Tests for type inference
├── test_data_loader.py          # Tests for data loading
├── test_analyzers/              # Tests for analysis modules
│   ├── __init__.py
│   ├── test_numeric.py          # Tests for numeric analysis
│   ├── test_categorical.py      # Tests for categorical analysis
│   └── test_text.py             # Tests for text analysis
├── test_visualizers/            # Tests for visualization modules
│   ├── __init__.py
│   ├── test_distribution.py     # Tests for distribution plots
│   └── test_relationship.py     # Tests for relationship plots
└── fixtures/                    # Test data files
    ├── sample_data.csv
    ├── text_data.csv
    └── config_samples/
        ├── default_config.yaml
        └── custom_config.yaml
```

## Running Tests

### Running the Full Test Suite

To run all tests with coverage reporting:

```bash
# From the project root directory
pytest

# With coverage reporting
pytest --cov=auto_eda --cov-report=term --cov-report=html
```

### Running Specific Tests

```bash
# Run a specific test file
pytest tests/test_core.py

# Run a specific test class
pytest tests/test_core.py::TestAutoEDA

# Run a specific test method
pytest tests/test_core.py::TestAutoEDA::test_load_data_from_csv

# Run tests matching a pattern
pytest -k "type_inference"
```

### Test Options

- `--verbose` or `-v`: Increase verbosity
- `--cov=auto_eda`: Enable coverage reporting for the auto_eda package
- `--cov-report=term`: Display coverage report in the terminal
- `--cov-report=html`: Generate HTML coverage report
- `--pdb`: Drop into debugger on test failure

## Test Fixtures

The `conftest.py` file contains shared fixtures used across multiple test files:

- `sample_dataframe`: A DataFrame with various column types for testing
- `sample_csv_path`: Path to a temporary CSV file with sample data
- `default_config`: Default configuration dictionary
- `config_file_path`: Path to a temporary configuration file
- `report_dir`: Path to a temporary directory for reports

## Adding New Tests

When adding new functionality to AutoEDA, please follow these guidelines for tests:

1. Create test files that match the module structure (e.g., `test_new_module.py`)
2. Use appropriate fixtures from `conftest.py` or create new ones if needed
3. Test both normal operation and edge cases/error conditions
4. Aim for high test coverage (>90%)
5. Include docstrings that explain what each test is verifying

## Test Data

The `fixtures/` directory contains sample data files used for testing. If you need to add new test data:

1. Keep file sizes small
2. Include a variety of data types and edge cases
3. Document the purpose and structure of the data

## Continuous Integration

Tests are automatically run on all pull requests and commits to the main branch. Pull requests cannot be merged unless all tests pass.
```

## How to Execute the Test Suite

Here are examples of how to execute the test suite:

### Basic Test Execution

```bash
# Navigate to the project root directory
cd auto-eda

# Run all tests
pytest

# Run with verbose output
pytest -v
```

### Running with Coverage

```bash
# Run tests with coverage reporting
pytest --cov=auto_eda

# Generate detailed HTML coverage report
pytest --cov=auto_eda --cov-report=html
# This creates a htmlcov/ directory with an interactive coverage report
```

### Running Specific Tests

```bash
# Run tests in a specific file
pytest tests/test_core.py

# Run a specific test class
pytest tests/test_core.py::TestAutoEDA

# Run a specific test method
pytest tests/test_core.py::TestAutoEDA::test_load_data_from_csv

# Run tests matching a keyword
pytest -k "type_inference"
```

### Debugging Tests

```bash
# Drop into debugger on test failure
pytest --pdb

# Show local variables in tracebacks
pytest --showlocals
```

### Generating Reports

```bash
# Generate JUnit XML report (useful for CI systems)
pytest --junitxml=test-results.xml

# Generate HTML report
pytest --html=report.html
```