import pandas as pd


# ==========================================================
# DERIVED KNOWLEDGE (RULE-BASED REASONING)
# ==========================================================

def derive_customer_spending(payments, orders):

    merged = payments.merge(orders, on="order_id")

    spending = (
        merged.groupby("customer_id")["payment_value"]
        .sum()
        .reset_index()
        .rename(columns={"payment_value": "total_spend"})
    )

    return spending


def derive_repeat_customers(orders):

    repeat = (
        orders.groupby("customer_id")["order_id"]
        .count()
        .reset_index()
        .rename(columns={"order_id": "order_count"})
    )

    repeat["is_repeat"] = repeat["order_count"] > 1

    return repeat


def derive_product_popularity(order_items):

    popularity = (
        order_items["product_id"]
        .value_counts()
        .reset_index()
    )

    popularity.columns = ["product_id", "purchase_count"]

    return popularity


# ==========================================================
# MAIN ANALYSIS
# ==========================================================

def run_analysis(customers, orders, order_items, products, payments):

    print("\n==============================")
    print(" KNOWLEDGE GRAPH ANALYTICS")
    print("==============================")

    # --------------------------------------------------
    # FACT 1
    # --------------------------------------------------

    total_revenue = payments["payment_value"].sum()

    print(f"\nTotal Revenue : {total_revenue:,.2f}")

    # --------------------------------------------------
    # FACT 2
    # --------------------------------------------------

    average_order_value = (
        payments.groupby("order_id")["payment_value"]
        .sum()
        .mean()
    )

    print(f"Average Order Value : {average_order_value:.2f}")

    # --------------------------------------------------
    # FACT 3
    # --------------------------------------------------

    total_orders = orders["order_id"].nunique()

    print(f"Total Orders : {total_orders}")

    # --------------------------------------------------
    # FACT 4
    # --------------------------------------------------

    total_customers = customers["customer_id"].nunique()

    print(f"Total Customers : {total_customers}")

    # --------------------------------------------------
    # FACT 5
    # --------------------------------------------------

    basket_size = (
        order_items.groupby("order_id")["product_id"]
        .count()
        .mean()
    )

    print(f"Average Basket Size : {basket_size:.2f}")

    # ==================================================
    # CUSTOMER LIFETIME VALUE
    # ==================================================

    customer_spending = derive_customer_spending(
        payments,
        orders
    )

    print("\n------------------------------")
    print("Top 10 Customers (CLV)")
    print("------------------------------")

    print(
        customer_spending
        .sort_values("total_spend", ascending=False)
        .head(10)
    )

    # ==================================================
    # REPEAT CUSTOMERS
    # ==================================================

    repeat = derive_repeat_customers(orders)

    repeat_count = repeat["is_repeat"].sum()

    print("\nRepeat Customers :", repeat_count)

    print("\nTop Repeat Customers")

    print(
        repeat
        .sort_values("order_count", ascending=False)
        .head(10)
    )

    # ==================================================
    # PRODUCT POPULARITY
    # ==================================================

    popularity = derive_product_popularity(order_items)

    print("\n------------------------------")
    print("Top Selling Products")
    print("------------------------------")

    print(popularity.head(10))

    # ==================================================
    # CATEGORY ANALYSIS
    # ==================================================

    merged = order_items.merge(
        products,
        on="product_id",
        how="left"
    )

    category_sales = (
        merged.groupby("product_category_name")
        .size()
        .reset_index(name="sales")
        .sort_values("sales", ascending=False)
    )

    print("\n------------------------------")
    print("Top Categories")
    print("------------------------------")

    print(category_sales.head(10))

    # ==================================================
    # MONTHLY REVENUE
    # ==================================================

    revenue = payments.merge(
        orders,
        on="order_id"
    )

    revenue["order_purchase_timestamp"] = pd.to_datetime(
        revenue["order_purchase_timestamp"]
    )

    revenue["Month"] = (
        revenue["order_purchase_timestamp"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly = (
        revenue.groupby("Month")["payment_value"]
        .sum()
        .reset_index()
    )

    print("\n------------------------------")
    print("Monthly Revenue")
    print("------------------------------")

    print(monthly.tail(12))

    # ==================================================
    # PAYMENT METHODS
    # ==================================================

    payment_stats = (
        payments["payment_type"]
        .value_counts()
        .reset_index()
    )

    payment_stats.columns = [
        "Payment Method",
        "Transactions"
    ]

    print("\n------------------------------")
    print("Payment Methods")
    print("------------------------------")

    print(payment_stats)

    # ==================================================
    # KNOWLEDGE GRAPH INTERPRETATION
    # ==================================================

    print("\n==============================")
    print(" KG BUSINESS INSIGHTS")
    print("==============================")

    print("• High CLV customers should be prioritised for retention.")

    print("• Frequently purchased products represent important graph hubs.")

    print("• Popular categories indicate strong demand.")

    print("• Repeat customers reveal long-term behavioural patterns.")

    print("• Payment preferences help understand customer purchasing behaviour.")

    print("• Monthly revenue trends can support business forecasting.")