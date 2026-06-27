import pandas as pd
from textblob import TextBlob

def run_sentiment_analysis(reviews):

    print("\n--- SENTIMENT ANALYSIS (REVIEWS) ---")

    # Clean data
    reviews = reviews.dropna(subset=["review_comment_message"]).copy()

    # Sentiment scoring
    reviews["sentiment"] = reviews["review_comment_message"].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity
    )

    print("\nAverage sentiment:", reviews["sentiment"].mean())

    print("\nMost positive reviews:")
    print(reviews.sort_values("sentiment", ascending=False)[["review_comment_message", "sentiment"]].head(5))

    print("\nMost negative reviews:")
    print(reviews.sort_values("sentiment")[["review_comment_message", "sentiment"]].head(5))