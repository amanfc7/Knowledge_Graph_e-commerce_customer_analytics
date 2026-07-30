from textblob import TextBlob
import pandas as pd
import os



# ----------------------------------------------------
# NLP SENTIMENT ANALYSIS
# REVIEW ENRICHMENT LAYER FOR KG
# ----------------------------------------------------

def run_sentiment_analysis(reviews):


    print("\n--- NLP LAYER (REVIEW SENTIMENT KG ENRICHMENT) ---")



    # ------------------------------------------------
    # Create results folder
    # ------------------------------------------------


    os.makedirs(

        "results",

        exist_ok=True

    )



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



    if len(reviews_clean) == 0:


        print(

            "No review text available"

        )


        return reviews_clean



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

        .apply(

            classify_sentiment

        )

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
    # Save Sentiment Dataset
    # ------------------------------------------------


    sentiment_file = (

        "results/"

        "review_sentiment_analysis.csv"

    )



    stats_file = (

        "results/"

        "sentiment_statistics.csv"

    )



    reviews_clean.to_csv(

        sentiment_file,

        index=False

    )



    sentiment_distribution.reset_index().to_csv(

        stats_file,

        index=False

    )



    print(

        "\nSentiment results saved:"

    )


    print(

        sentiment_file

    )


    print(

        stats_file

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
    into numerical properties that enrich
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



    Sentiment supports:

        - customer satisfaction analysis
        - product quality analysis
        - recommendation improvements
        - seller reputation analysis


    Future KG enrichment:

    REVIEW
       |
       HAS_SENTIMENT
       |
    SENTIMENT_CONCEPT

    """

    )



    return reviews_clean