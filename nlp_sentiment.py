from textblob import TextBlob


def run_sentiment_analysis(reviews):

    print("\n--- NLP LAYER (REVIEW SENTIMENT KG ENRICHMENT) ---")

    reviews = reviews.dropna(subset=["review_comment_message"]).copy()

    # -------------------------
    # SENTIMENT AS DERIVED PROPERTY
    # -------------------------
    reviews["sentiment"] = reviews["review_comment_message"].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity
    )

    print("\nDERIVED FACT: Average Sentiment =", reviews["sentiment"].mean())

    print("\nTop Positive Reviews:")
    print(
        reviews.sort_values("sentiment", ascending=False)[
            ["review_comment_message", "sentiment"]
        ].head(5)
    )

    print("\nTop Negative Reviews:")
    print(
        reviews.sort_values("sentiment")[
            ["review_comment_message", "sentiment"]
        ].head(5)
    )

    print("\n--- INTERPRETATION ---")
    print("Sentiment acts as a soft signal enriching the Knowledge Graph nodes.")