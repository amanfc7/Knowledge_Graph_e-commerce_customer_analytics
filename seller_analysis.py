import pandas as pd

def run_seller_analysis(order_items, payments):

    print("\n--- SELLER PERFORMANCE ANALYSIS ---")

    # Merge seller revenue
    merged = order_items.merge(payments, on="order_id")

    seller_revenue = merged.groupby("seller_id")["payment_value"].sum().sort_values(ascending=False)

    print("\nTop Sellers by Revenue:")
    print(seller_revenue.head(10))

    # Product diversity per seller
    seller_diversity = order_items.groupby("seller_id")["product_id"].nunique().sort_values(ascending=False)

    print("\nTop Sellers by Product Diversity:")
    print(seller_diversity.head(10))