import pandas as pd


# ----------------------------------------------------
# SELLER PERFORMANCE ANALYSIS
# KG ECONOMIC LAYER
# ----------------------------------------------------

def run_seller_analysis(order_items, payments):

    print("\n--- SELLER KG ANALYSIS (ECONOMIC RELATIONAL VIEW) ---")


    # ------------------------------------------------
    # Merge transaction + payment information
    # ------------------------------------------------

    merged = order_items.merge(
        payments,
        on="order_id",
        how="inner"
    )


    # ------------------------------------------------
    # 1. Seller Revenue Analysis
    # ------------------------------------------------

    seller_revenue = (
        merged
        .groupby("seller_id")["payment_value"]
        .sum()
        .sort_values(
            ascending=False
        )
    )


    print("\nDERIVED FACT: Seller Revenue Ranking")

    print(
        seller_revenue.head(10)
    )


    # ------------------------------------------------
    # 2. Seller Transaction Volume
    # ------------------------------------------------

    seller_transactions = (
        merged
        .groupby("seller_id")["order_id"]
        .nunique()
        .sort_values(
            ascending=False
        )
    )


    print("\nDERIVED FACT: Seller Transaction Volume")

    print(
        seller_transactions.head(10)
    )



    # ------------------------------------------------
    # 3. Seller Product Diversity
    # ------------------------------------------------

    seller_diversity = (
        order_items
        .groupby("seller_id")["product_id"]
        .nunique()
        .sort_values(
            ascending=False
        )
    )


    print(
        "\nDERIVED FACT: Seller Product Diversity"
    )

    print(
        seller_diversity.head(10)
    )



    # ------------------------------------------------
    # 4. Economic Importance Score
    # ------------------------------------------------
    # Combines:
    # - revenue
    # - transaction activity
    # - product variety
    #
    # This represents a KG derived feature
    # ------------------------------------------------


    seller_score = pd.DataFrame({

        "revenue":
        seller_revenue,


        "transactions":
        seller_transactions,


        "product_diversity":
        seller_diversity

    }).fillna(0)



    # Normalized importance score

    seller_score["economic_score"] = (

        seller_score["revenue"]
        /
        seller_score["revenue"].max()

        +

        seller_score["transactions"]
        /
        seller_score["transactions"].max()

        +

        seller_score["product_diversity"]
        /
        seller_score["product_diversity"].max()

    ) / 3



    print(
        "\nDERIVED FACT: Seller Economic Centrality"
    )


    print(
        seller_score
        .sort_values(
            "economic_score",
            ascending=False
        )
        .head(10)
    )



    # ------------------------------------------------
    # KG Interpretation
    # ------------------------------------------------

    print(
        "\n--- KG BUSINESS INTERPRETATION ---"
    )


    print(
        """
    Seller nodes are important economic entities.

    High revenue sellers:
        -> financially central nodes

    High diversity sellers:
        -> connected to many product entities

    High economic score sellers:
        -> influential nodes in the commerce graph
    """
    )


    return seller_score