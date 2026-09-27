import os

import pandas as pd
from textblob import TextBlob


# nlp sentiment analysis
# review enrichment layer for the KG

NLP_SENTIMENT_RELATION = "HAS_NLP_SENTIMENT"
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


def classify_sentiment(score):
    """Classify TextBlob polarity using project-defined thresholds."""
    if score > POSITIVE_THRESHOLD:
        return "positive"
    if score < NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def run_sentiment_analysis(reviews, G):
    """Analyse review text and add NLP-derived knowledge to the KG."""
    print("\n--- NLP LAYER (REVIEW SENTIMENT KG ENRICHMENT) ---")
    os.makedirs("results", exist_ok=True)

    required_column = "review_comment_message"
    if required_column not in reviews.columns:
        print(f"Missing required column: {required_column}")
        return pd.DataFrame()

    # data cleaning
    reviews_clean = reviews.dropna(
        subset=[required_column]
    ).copy()
    reviews_clean[required_column] = (
        reviews_clean[required_column].astype(str).str.strip()
    )
    reviews_clean = reviews_clean[
        reviews_clean[required_column] != ""
    ].copy()

    print("Reviews analysed:", len(reviews_clean))

    if reviews_clean.empty:
        print("No review text available.")
        return reviews_clean

    # sentiment extraction
    reviews_clean["sentiment"] = reviews_clean[
        required_column
    ].apply(
        lambda text: TextBlob(text).sentiment.polarity
    )

    # sentiment classification
    reviews_clean["sentiment_label"] = (
        reviews_clean["sentiment"].apply(classify_sentiment)
    )

    # derived statistics
    average_sentiment = reviews_clean["sentiment"].mean()
    sentiment_distribution = reviews_clean["sentiment_label"].value_counts()

    print(
        "\nDERIVED FACT: Average TextBlob Sentiment =",
        round(average_sentiment, 4),
    )
    print("\nDERIVED FACT: Sentiment Distribution")
    print(sentiment_distribution)

    # evolve knowledge graph
    print("\n--- EVOLVING KNOWLEDGE GRAPH ---")

    evolution_records = []
    added_nodes = 0
    added_edges = 0

    for review_index, row in reviews_clean.iterrows():
        review_node = f"REVIEW_{review_index}"

        if review_node not in G:
            continue

        sentiment_label = row["sentiment_label"]
        sentiment_score = float(row["sentiment"])

        # add NLP-derived properties to the review entity
        G.nodes[review_node]["nlp_polarity"] = sentiment_score
        G.nodes[review_node]["nlp_sentiment"] = sentiment_label
        G.nodes[review_node]["nlp_source"] = "TextBlob"
        G.nodes[review_node]["nlp_thresholds"] = (
            f"{NEGATIVE_THRESHOLD}/{POSITIVE_THRESHOLD}"
        )

        # create one reusable NLP sentiment concept per label
        sentiment_node = f"SENTIMENT_NLP_{sentiment_label}"

        if sentiment_node not in G:
            G.add_node(
                sentiment_node,
                type="sentiment",
                schema_type="concept",
                source="TextBlob_NLP",
                display_name=f"NLP Sentiment | {sentiment_label}",
            )
            added_nodes += 1

        # add NLP-derived relationship
        edge_exists = G.has_edge(review_node, sentiment_node)

        if not edge_exists:
            G.add_edge(
                review_node,
                sentiment_node,
                relation=NLP_SENTIMENT_RELATION,
                source="TextBlob_NLP",
            )
            added_edges += 1
            action = "added"
        else:
            action = "updated"

        evolution_records.append({
            "review_node": review_node,
            "sentiment_node": sentiment_node,
            "sentiment_label": sentiment_label,
            "polarity": sentiment_score,
            "relation": NLP_SENTIMENT_RELATION,
            "source": "TextBlob",
            "action": action,
        })

    print("Reviews enriched:", len(evolution_records))
    print("NLP sentiment nodes added:", added_nodes)
    print("NLP sentiment edges added:", added_edges)
    print("KG nodes:", G.number_of_nodes())
    print("KG edges:", G.number_of_edges())

    # save KG evolution log
    evolution_file = "results/kg_sentiment_evolution.csv"
    pd.DataFrame(evolution_records).to_csv(
        evolution_file,
        index=False,
    )

    print("\nKG evolution log saved:")
    print(evolution_file)

    # save sentiment dataset and statistics
    sentiment_file = "results/review_sentiment_analysis.csv"
    stats_file = "results/sentiment_statistics.csv"

    reviews_clean.to_csv(sentiment_file, index=False)

    sentiment_distribution.rename_axis("sentiment").reset_index(
        name="review_count"
    ).to_csv(
        stats_file,
        index=False,
    )

    print("\nSentiment results saved:")
    print(sentiment_file)
    print(stats_file)

    # display representative positive and negative reviews
    print("\nTop Positive Reviews:")
    print(
        reviews_clean.sort_values(
            "sentiment",
            ascending=False,
        )[
            [required_column, "sentiment", "sentiment_label"]
        ].head(5)
    )

    print("\nTop Negative Reviews:")
    print(
        reviews_clean.sort_values(
            "sentiment",
            ascending=True,
        )[
            [required_column, "sentiment", "sentiment_label"]
        ].head(5)
    )

    # KG interpretation
    print("\n--- KG INTERPRETATION ---")
    print(
        """
Review text provides unstructured information that can be
converted into structured knowledge.

The NLP layer adds:
    - TextBlob polarity as a numeric property
    - an NLP sentiment label
    - HAS_NLP_SENTIMENT relationships
    - provenance identifying TextBlob as the source

The original review-score sentiment remains separate.

This distinction allows the KG to represent:
    - sentiment derived from review score
    - sentiment derived from review text

The enriched KG can support exploratory analysis of:
    - customer satisfaction
    - product reviews
    - seller reputation
    - review-text patterns
    """
    )

    return reviews_clean