def run_analysis(customers, orders, order_items, products, payments):

    print("\n--- BASIC ANALYSIS (LO9) ---")

    # Revenue
    total_revenue = payments["payment_value"].sum()
    print("Total Revenue:", total_revenue)

    # Top customers
    customer_orders = customers.merge(orders, on="customer_id")
    top_customers = customer_orders["customer_id"].value_counts().head(10)
    print("\nTop Customers:\n", top_customers)

    # Top products
    top_products = order_items["product_id"].value_counts().head(10)
    print("\nTop Products:\n", top_products)

    # Category analysis
    merged = order_items.merge(products, on="product_id")
    top_categories = merged["product_category_name"].value_counts().head(10)
    print("\nTop Categories:\n", top_categories)