import pandas as pd

# -----------------------------
# KG REASONING LAYER (DATALOG-STYLE)
# -----------------------------

def derive_customer_spending(payments, orders):
    """
    Derived fact: CustomerTotalSpend(customer_id, total_value)
    """

    merged = payments.merge(orders, on="order_id")

    spending = merged.groupby("customer_id")["payment_value"].sum().reset_index()
    spending.columns = ["customer_id", "total_spend"]

    return spending


def derive_repeat_customers(orders):
    """
    Derived fact: RepeatCustomer(customer_id)
    """

    freq = orders.groupby("customer_id")["order_id"].count().reset_index()
    freq.columns = ["customer_id", "order_count"]

    freq["is_repeat"] = freq["order_count"] > 1

    return freq


def derive_top_products(order_items):
    """
    Derived fact: ProductPopularity(product_id, count)
    """

    top = order_items["product_id"].value_counts().reset_index()
    top.columns = ["product_id", "purchase_count"]

    return top


# -----------------------------
# MAIN ANALYSIS PIPELINE
# -----------------------------

def run_analysis(customers, orders, order_items, products, payments):

    print("\n--- KG REASONING + ANALYTICS LAYER ---")

    # -------------------------
    # FACT 1: TOTAL REVENUE
    # -------------------------
    total_revenue = payments["payment_value"].sum()
    print("FACT: Total Revenue =", total_revenue)

    # -------------------------
    # DERIVED FACT: CUSTOMER SPENDING
    # -------------------------
    customer_spend = derive_customer_spending(payments, orders)

    print("\nDERIVED FACT: Top Customer Spend")
    print(customer_spend.sort_values("total_spend", ascending=False).head(10))

    # -------------------------
    # DERIVED FACT: REPEAT CUSTOMERS
    # -------------------------
    repeat = derive_repeat_customers(orders)

    print("\nDERIVED FACT: Repeat Customers")
    print(repeat.sort_values("order_count", ascending=False).head(10))

    # -------------------------
    # DERIVED FACT: PRODUCT POPULARITY
    # -------------------------
    product_pop = derive_top_products(order_items)

    print("\nDERIVED FACT: Top Products")
    print(product_pop.head(10))

    # -------------------------
    # CATEGORY INSIGHT (LIGHT WEIGHT REASONING)
    # -------------------------
    merged = order_items.merge(products, on="product_id", how="left")

    category_stats = merged.groupby("product_category_name").size().reset_index()
    category_stats.columns = ["category", "count"]

    print("\nDERIVED FACT: Top Categories")
    print(category_stats.sort_values("count", ascending=False).head(10))