from textblob import TextBlob
import pandas as pd



# ----------------------------------------------------
# NLP SENTIMENT ANALYSIS
# REVIEW ENRICHMENT LAYER FOR KG
# ----------------------------------------------------

def run_sentiment_analysis(reviews):

    print("\n--- NLP LAYER (REVIEW SENTIMENT KG ENRICHMENT) ---")


    # ------------------------------------------------
    # Data Cleaning
    # ------------------------------------------------

    reviews_clean = reviews.dropna(
        subset=[
            "review_comment_message"
        ]
    ).copy()



    print(
        "Reviews analysed:",
        len(reviews_clean)
    )



    # ------------------------------------------------
    # Sentiment Extraction
    # ------------------------------------------------

    reviews_clean["sentiment"] = (

        reviews_clean[
            "review_comment_message"
        ]

        .apply(
            lambda x:
            TextBlob(
                str(x)
            )
            .sentiment
            .polarity
        )

    )



    # ------------------------------------------------
    # Sentiment Classification
    # ------------------------------------------------

    def classify_sentiment(score):

        if score > 0.05:
            return "positive"

        elif score < -0.05:
            return "negative"

        else:
            return "neutral"



    reviews_clean["sentiment_label"] = (

        reviews_clean["sentiment"]
        .apply(classify_sentiment)

    )



    # ------------------------------------------------
    # KG Derived Statistics
    # ------------------------------------------------

    average_sentiment = (
        reviews_clean["sentiment"]
        .mean()
    )


    sentiment_distribution = (

        reviews_clean["sentiment_label"]
        .value_counts()

    )



    print(
        "\nDERIVED FACT: Average Sentiment =",
        round(
            average_sentiment,
            4
        )
    )


    print(
        "\nDERIVED FACT: Sentiment Distribution"
    )


    print(
        sentiment_distribution
    )



    # ------------------------------------------------
    # Positive Reviews
    # ------------------------------------------------

    print(
        "\nTop Positive Reviews:"
    )


    print(

        reviews_clean
        .sort_values(
            "sentiment",
            ascending=False
        )
        [
            [
                "review_comment_message",
                "sentiment"
            ]
        ]
        .head(5)

    )



    # ------------------------------------------------
    # Negative Reviews
    # ------------------------------------------------

    print(
        "\nTop Negative Reviews:"
    )


    print(

        reviews_clean
        .sort_values(
            "sentiment"
        )
        [
            [
                "review_comment_message",
                "sentiment"
            ]
        ]
        .head(5)

    )



    # ------------------------------------------------
    # KG Interpretation
    # ------------------------------------------------

    print(
        "\n--- KG INTERPRETATION ---"
    )


    print(
        """
    Reviews are unstructured external knowledge.

    Sentiment converts textual information
    into numerical properties that can enrich
    KG entities.

    Example:

    Customer
        |
        PURCHASED
        |
    Product
        |
        HAS_REVIEW
        |
    Sentiment Score


    Sentiment can support:
        - customer satisfaction analysis
        - product quality analysis
        - recommendation improvements
    """
    )


    return reviews_clean