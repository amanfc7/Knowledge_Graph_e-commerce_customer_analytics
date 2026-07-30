import pandas as pd
import os



# ----------------------------------------------------
# SELLER PERFORMANCE ANALYSIS
# KG ECONOMIC LAYER
# ----------------------------------------------------

def run_seller_analysis(order_items, payments):


    print("\n--- SELLER KG ANALYSIS (ECONOMIC RELATIONAL VIEW) ---")


    # ------------------------------------------------
    # Create results folder
    # ------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True
    )



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
    # 4. Average Order Value Per Seller
    # ------------------------------------------------


    seller_aov = (

        merged

        .groupby("seller_id")["payment_value"]

        .mean()

        .sort_values(
            ascending=False
        )

    )



    print(
        "\nDERIVED FACT: Seller Average Order Value"
    )


    print(
        seller_aov.head(10)
    )



    # ------------------------------------------------
    # 5. Economic Importance Score
    # ------------------------------------------------
    #
    # Combines:
    #
    # - revenue
    # - transaction activity
    # - product variety
    # - average order value
    #
    # Represents KG derived importance
    # ------------------------------------------------



    seller_score = pd.DataFrame({

        "revenue":
        seller_revenue,


        "transactions":
        seller_transactions,


        "product_diversity":
        seller_diversity,


        "average_order_value":
        seller_aov

    }).fillna(0)



    # ------------------------------------------------
    # Normalization
    # ------------------------------------------------


    def normalize(column):

        maximum = column.max()

        if maximum == 0:

            return column

        return column / maximum



    seller_score["economic_score"] = (

        normalize(
            seller_score["revenue"]
        )

        +

        normalize(
            seller_score["transactions"]
        )

        +

        normalize(
            seller_score["product_diversity"]
        )

        +

        normalize(
            seller_score["average_order_value"]
        )

    ) / 4



    # ------------------------------------------------
    # Ranking
    # ------------------------------------------------


    seller_score = (

        seller_score

        .sort_values(
            "economic_score",
            ascending=False
        )

    )


    seller_score["rank"] = range(
        1,
        len(seller_score)+1
    )



    print(
        "\nDERIVED FACT: Seller Economic Centrality"
    )


    print(
        seller_score.head(10)
    )



    # ------------------------------------------------
    # Save Results
    # ------------------------------------------------


    output_file = (
        "results/"
        "seller_performance_analysis.csv"
    )


    seller_score.to_csv(
        output_file
    )


    print(
        "\nSeller analysis saved:",
        output_file
    )



    # ------------------------------------------------
    # KG Business Interpretation
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
        -> influential nodes in commerce graph

    These metrics can later power:
        - Streamlit dashboards
        - seller ranking views
        - anomaly detection
        - recommendation systems
    """
    )



    return seller_score