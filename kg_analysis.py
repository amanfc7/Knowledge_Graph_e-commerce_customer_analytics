def run_analysis(customers, orders, order_items, products, payments):

    print("\n--- ADVANCED ANALYTICS (LO9 + EXCEED) ---")

    # -------------------------
    # Revenue
    # -------------------------
    total_revenue = payments["payment_value"].sum()
    print("Total Revenue:", total_revenue)

    # -------------------------
    # Customer Lifetime Value (CLV)
    # -------------------------
    customer_clv = payments.merge(orders, on="order_id") \
        .groupby("customer_id")["payment_value"] \
        .sum() \
        .sort_values(ascending=False) \
        .head(10)

    print("\nTop Customers (CLV):\n", customer_clv)

    # -------------------------
    # Repeat customers (behavior)
    # -------------------------
    repeat_customers = customers.merge(orders, on="customer_id") \
        .groupby("customer_id")["order_id"].count() \
        .sort_values(ascending=False) \
        .head(10)

    print("\nRepeat Purchase Behavior:\n", repeat_customers)

    # -------------------------
    # Top Products
    # -------------------------
    top_products = order_items["product_id"].value_counts().head(10)
    print("\nTop Products:\n", top_products)

    # -------------------------
    # Category revenue insight
    # -------------------------
    merged = order_items.merge(products, on="product_id")
    top_categories = merged["product_category_name"].value_counts().head(10)
    print("\nTop Categories:\n", top_categories)