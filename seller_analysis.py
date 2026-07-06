import pandas as pd


def run_seller_analysis(order_items, payments):

    print("\n--- SELLER KG ANALYSIS (RELATIONAL VIEW) ---")

    # -------------------------
    # SELLER REVENUE GRAPH EDGE
    # -------------------------
    merged = order_items.merge(payments, on="order_id")

    seller_revenue = merged.groupby("seller_id")["payment_value"].sum().sort_values(ascending=False)

    print("\nDERIVED FACT: Seller Revenue (Top 10)")
    print(seller_revenue.head(10))

    # -------------------------
    # SELLER SPECIALIZATION (KG PROPERTY)
    # -------------------------
    seller_diversity = order_items.groupby("seller_id")["product_id"].nunique().sort_values(ascending=False)

    print("\nDERIVED FACT: Seller Product Diversity")
    print(seller_diversity.head(10))

    # -------------------------
    # INTERPRETATION LAYER (KG INSIGHT)
    # -------------------------
    print("\n--- KG INSIGHT ---")
    print("High diversity sellers ≈ generalists in product graph space")
    print("High revenue sellers ≈ central economic nodes in KG")