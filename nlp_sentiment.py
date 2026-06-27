from textblob import TextBlob

def analyze_sentiment(reviews_df):

    print("\n--- SENTIMENT ANALYSIS (EXCEED LO) ---")

    reviews_df = reviews_df.copy()

    reviews_df["sentiment"] = reviews_df["review_comment_message"].astype(str).apply(
        lambda x: TextBlob(x).sentiment.polarity
    )

    print("\nAverage Sentiment:", reviews_df["sentiment"].mean())

    print("\nTop Positive Reviews:")
    print(reviews_df.sort_values("sentiment", ascending=False).head(5))