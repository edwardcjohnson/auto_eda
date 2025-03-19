import pytest
import pandas as pd
import numpy as np
from auto_eda.analyzers.categorical import (
    analyze_categorical_columns,
    calculate_entropy,
    analyze_categorical_relationships,
)

# Fixture with a sample DataFrame to test basic analysis and relationships.
@pytest.fixture
def sample_df():
    data = {
        'Category_A': ['A', 'B', 'A', 'C', 'A', 'B', 'C', 'A'],
        'Category_B': ['X', 'Y', 'X', 'Z', 'X', 'Y', 'Z', 'X'],
        'Target':     ['T1', 'T2', 'T1', 'T2', 'T1', 'T1', 'T2', 'T2']
    }
    return pd.DataFrame(data)

def test_analyze_categorical_columns_basic(sample_df):
    # Analyze both Category_A and Category_B
    result = analyze_categorical_columns(sample_df, columns=['Category_A', 'Category_B'])
    assert 'Category_A' in result
    assert 'Category_B' in result
    
    cat_a_stats = result['Category_A']
    cat_b_stats = result['Category_B']
    
    # For Category_A, unique values should be 3 (A, B, C)
    assert cat_a_stats['unique_count'] == 3
    # Expect no missing values in the sample
    assert cat_a_stats['missing_count'] == 0
    # Mode of Category_A should be 'A' since it appears 4 times.
    assert cat_a_stats['mode'] == 'A'
    # Mode count should match the frequency count of 'A'
    assert cat_a_stats['mode_count'] == 4
    # Frequency table should be a list of dictionaries
    assert isinstance(cat_a_stats['frequency_table'], list)

def test_calculate_entropy():
    # Uniform distribution: four classes each with probability 0.25 should yield entropy 2.0.
    probabilities = pd.Series([0.25, 0.25, 0.25, 0.25])
    ent = calculate_entropy(probabilities)
    assert abs(ent - 2.0) < 1e-6

    # Non-uniform distribution
    probabilities = pd.Series([0.4, 0.3, 0.2, 0.1])
    expected_entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    ent = calculate_entropy(probabilities)
    assert abs(ent - expected_entropy) < 1e-6

def test_analyze_categorical_max_categories():
    # Create a DataFrame with more than a few categories.
    data = {
        "Cat": list("ABCDEFGHIJK")  # 11 distinct categories
    }
    df = pd.DataFrame(data)
    # Set max_categories = 5 => should return top 4 categories and group the rest as "Other".
    result = analyze_categorical_columns(df, columns=["Cat"], max_categories=5)
    stats = result["Cat"]
    frequency_table = stats["frequency_table"]
    # Expecting 5 rows in frequency table.
    assert len(frequency_table) == 5
    # Last row should be the aggregated "Other" category.
    assert frequency_table[-1]["value"] == "Other"

def test_analyze_categorical_missing_values():
    # Create a DataFrame where the column has missing (NaN) values and empty strings.
    data = {
        "Cats": ["A", None, "", "B", "A", None, "C", "A"]
    }
    df = pd.DataFrame(data)
    result = analyze_categorical_columns(df, columns=["Cats"])
    stats = result["Cats"]
    # Typically, None values count as missing; empty strings are _not_ NaN.
    assert stats["missing_count"] == 2
    # Unique count should include "", "A", "B", "C" (four distinct values).
    assert stats["unique_count"] == 4

def test_analyze_categorical_relationships(sample_df):
    # Use 'Target' as the target variable and analyze the relationships for Category_A and Category_B
    rel_result = analyze_categorical_relationships(
        sample_df,
        target_column="Target",
        categorical_columns=["Category_A", "Category_B"]
    )
    # Expect results for both columns (if target exists).
    assert "Category_A" in rel_result
    assert "Category_B" in rel_result
    
    rel_a = rel_result["Category_A"]
    rel_b = rel_result["Category_B"]
    
    # Check that contingency tables are lists
    assert isinstance(rel_a["contingency_table"], list)
    assert isinstance(rel_b["contingency_table"], list)
    
    # If chi-square test was performed, it should have the expected keys.
    if rel_a["chi2_test"] is not None:
        for key in ["chi2", "p_value", "degrees_of_freedom"]:
            assert key in rel_a["chi2_test"]
