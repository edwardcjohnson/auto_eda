#!/usr/bin/env python3
"""
Text Analysis Example for AutoEDA

This script creates a sample DataFrame with a text column (e.g., product reviews)
and demonstrates text analysis and sentiment analysis using the AutoEDA text analyzer.
"""

import pandas as pd
from auto_eda.analyzers.text import analyze_text_columns, perform_sentiment_analysis

def main():
    # Create a sample DataFrame with 20 rows and 2 columns:
    # one containing reviews and one for a predefined sentiment category label.
    data = {
        "review": [
            "I love this product! It works exceptionally well and exceeds my expectations.",
            "Terrible service. I will never buy from this company again.",
            "Quite mediocre, neither impressive nor disappointing.",
            "Absolutely fantastic! Highly recommended.",
            "It did what it was supposed to do, nothing more, nothing less.",
            "Worst purchase I've ever made.",
            "I am extremely satisfied with the quality and performance.",
            "Not worth the money. Very disappointing.",
            "Excellent value! Great customer service and quality.",
            "The product broke within a week. Very poor build quality.",
            "I absolutely love it, five stars! Will re-purchase.",
            "The design is sleek and modern, but the performance is lacking.",
            "Awful! It stopped working after a few uses.",
            "Simply superb, a top‐notch experience.",
            "Good, but could be improved in future models.",
            "The best I have seen under this price range.",
            "Subpar, not very impressive.",
            "An average product with acceptable performance.",
            "Highly inefficient and overrated.",
            "Outstanding! Exceeded all expectations."
        ],
        "sentiment_category": [
            "Positive", "Negative", "Neutral", "Positive", "Neutral",
            "Negative", "Positive", "Negative", "Positive", "Negative",
            "Positive", "Neutral", "Negative", "Positive", "Positive",
            "Positive", "Neutral", "Neutral", "Negative", "Positive"
        ]
    }
    df = pd.DataFrame(data)
    
    # Analyze the text column 'review'
    text_analysis_results = analyze_text_columns(df, columns=["review"], max_words=50)
    
    # Perform sentiment analysis on the 'review' column.
    sentiment_analysis_results = perform_sentiment_analysis(df, text_columns=["review"])
    
    # Print the results to console
    print("Text Analysis Results:")
    for col, stats in text_analysis_results.items():
        print(f"\nColumn: {col}")
        for key, value in stats.items():
            if key != "word_frequency":
                print(f"  {key}: {value}")
            else:
                print("  word_frequency:")
                for record in value:
                    print(f"    {record}")
    
    print("\nSentiment Analysis Results:")
    for col, stats in sentiment_analysis_results.items():
        print(f"\nColumn: {col}")
        for key, value in stats.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()