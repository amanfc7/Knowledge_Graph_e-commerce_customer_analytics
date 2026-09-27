import os
from datetime import datetime

import pandas as pd

# result storage
RESULT_FOLDER = "results"


def save_result(df, filename):
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    path = os.path.join(RESULT_FOLDER, filename)
    df.to_csv(path, index=False)
    print("Saved result:", path)


# derived knowledge and rule-based analysis
def derive_customer_spending(payments, orders, customers):
    """
    Calculate historical customer spending using customer_unique_id.

    customer_id identifies the customer record associated with an order,
    while customer_unique_id identifies the persistent customer across
    multiple orders.

    """
    order_customer = orders[["order_id", "customer_id"]].merge(
        customers[["customer_id", "customer_unique_id"]],
        on="customer_id",
        how="left",
    )

    spending = payments.merge(
        order_customer,
        on="order_id",
        how="left",
    )

    customer_spending = (
        spending
        .groupby("customer_unique_id", as_index=False)
        .agg(
            total_spending=("payment_value", "sum"),
            order_count=("order_id", "nunique"),
        )
    )

    customer_spending["average_order_value"] = (
        customer_spending["total_spending"]
        / customer_spending["order_count"].replace(0, pd.NA)
    )

    return customer_spending.sort_values(
        "total_spending",
        ascending=False,
    )


def derive_repeat_customers(orders, customers):
    """Identify persistent customers with more than one distinct order."""
    order_customer = orders[["order_id", "customer_id"]].merge(
        customers[["customer_id", "customer_unique_id"]],
        on="customer_id",
        how="left",
    )

    customer_orders = (
        order_customer
        .dropna(subset=["customer_unique_id"])
        .groupby("customer_unique_id", as_index=False)
        .agg(order_count=("order_id", "nunique"))
    )

    customer_orders["is_repeat"] = customer_orders["order_count"] > 1
    return customer_orders


def validate_customer_identity(customers, orders):
    """Validate customer_id and customer_unique_id relationships."""
    customer_record_count = customers["customer_id"].nunique()
    unique_customer_count = customers["customer_unique_id"].nunique()

    order_customer = orders[["order_id", "customer_id"]].merge(
        customers[["customer_id", "customer_unique_id"]],
        on="customer_id",
        how="left",
    )

    missing_customer_identity = int(
        order_customer["customer_unique_id"].isna().sum()
    )

    customer_orders = (
        order_customer
        .dropna(subset=["customer_unique_id"])
        .groupby("customer_unique_id")["order_id"]
        .nunique()
    )

    repeat_customer_count = int((customer_orders > 1).sum())

    validation = {
        "customer_records": customer_record_count,
        "unique_persistent_customers": unique_customer_count,
        "orders": orders["order_id"].nunique(),
        "orders_without_customer_identity": missing_customer_identity,
        "repeat_customers": repeat_customer_count,
    }

    print("\n==============================")
    print(" CUSTOMER IDENTITY VALIDATION")
    print("==============================")
    print(f"Customer Records : {customer_record_count:,}")
    print(f"Unique Customers : {unique_customer_count:,}")
    print(f"Orders : {validation['orders']:,}")
    print(
        "Orders Without Customer Identity : "
        f"{missing_customer_identity:,}"
    )
    print(f"Repeat Customers : {repeat_customer_count:,}")

    return validation


def derive_product_popularity(order_items):
    """Count order-item associations for each product."""
    popularity = order_items["product_id"].value_counts().reset_index()
    popularity.columns = ["product_id", "purchase_count"]
    return popularity


def derive_financial_kg_profiles(
    payments,
    orders,
    customers,
    order_items,
    products,
):
    """
    Build a KG-oriented financial customer profile.

    The analysis connects customer, order, payment, product and category
    information to show how financial behaviour can be analysed together
    with product relationships.

    Payment values remain attached to orders/customers and are not
    redistributed across categories or products.
    """
    order_customer = orders[["order_id", "customer_id"]].merge(
        customers[["customer_id", "customer_unique_id"]],
        on="customer_id",
        how="left",
    )

    customer_payments = payments.merge(
        order_customer,
        on="order_id",
        how="left",
    )

    financial_profile = (
        customer_payments
        .groupby("customer_unique_id", as_index=False)
        .agg(
            total_payment_value=("payment_value", "sum"),
            order_count=("order_id", "nunique"),
        )
    )

    order_products = order_items[["order_id", "product_id"]].merge(
        products[["product_id", "product_category_name_english"]],
        on="product_id",
        how="left",
    )

    customer_products = order_customer.merge(
        order_products,
        on="order_id",
        how="inner",
    )

    product_profile = (
        customer_products
        .groupby("customer_unique_id", as_index=False)
        .agg(
            distinct_products=("product_id", "nunique"),
            distinct_categories=(
                "product_category_name_english",
                "nunique",
            ),
        )
    )

    profile = financial_profile.merge(
        product_profile,
        on="customer_unique_id",
        how="left",
    )

    profile["average_payment_per_order"] = (
        profile["total_payment_value"]
        / profile["order_count"].replace(0, pd.NA)
    )

    profile["distinct_products"] = profile["distinct_products"].fillna(0)
    profile["distinct_categories"] = profile["distinct_categories"].fillna(0)

    profile = profile.sort_values(
        "total_payment_value",
        ascending=False,
    )

    return profile


# main analysis
def run_analysis(
    customers,
    orders,
    order_items,
    products,
    payments,
):
    print("\n==============================")
    print(" KNOWLEDGE GRAPH ANALYTICS")
    print("==============================")

    os.makedirs(RESULT_FOLDER, exist_ok=True)

    metadata = {
        "execution_time": str(datetime.now()),
        "customers": len(customers),
        "orders": len(orders),
        "products": len(products),
    }

    pd.DataFrame([metadata]).to_csv(
        os.path.join(RESULT_FOLDER, "run_metadata.csv"),
        index=False,
    )

    # overall payment value
    total_payment_value = payments["payment_value"].sum()
    print(f"\nTotal Payment Value : {total_payment_value:,.2f}")

    # average payment value per order
    order_payment_values = payments.groupby("order_id")["payment_value"].sum()
    average_order_value = order_payment_values.mean()
    print(f"Average Order Value : {average_order_value:.2f}")

    # total orders
    total_orders = orders["order_id"].nunique()
    print(f"Total Orders : {total_orders:,}")

    # customer counts
    customer_record_count = customers["customer_id"].nunique()
    unique_customer_count = customers["customer_unique_id"].nunique()

    print(f"Customer Records : {customer_record_count:,}")
    print(f"Unique Customers : {unique_customer_count:,}")

    # average basket size
    basket_size = (
        order_items
        .groupby("order_id")["product_id"]
        .count()
        .mean()
    )

    print(f"Average Basket Size : {basket_size:.2f}")

    # historical customer spending
    customer_spending = derive_customer_spending(
        payments,
        orders,
        customers,
    )

    # keep filename for pipeline compatibility
    save_result(customer_spending, "customer_clv.csv")

    print("\n------------------------------")
    print("Top 10 Customers by Historical Spending")
    print("------------------------------")
    print(customer_spending.head(10))

    # customer identity validation
    identity_validation = validate_customer_identity(
        customers,
        orders,
    )

    pd.DataFrame([identity_validation]).to_csv(
        os.path.join(
            RESULT_FOLDER,
            "customer_identity_validation.csv",
        ),
        index=False,
    )

    # repeat customers
    repeat = derive_repeat_customers(
        orders,
        customers,
    )

    save_result(repeat, "repeat_customers.csv")

    repeat_count = int(repeat["is_repeat"].sum())
    print("\nRepeat Customers:", repeat_count)

    # product popularity
    popularity = derive_product_popularity(order_items)

    save_result(popularity, "product_popularity.csv")

    print("\n------------------------------")
    print("Top Selling Products")
    print("------------------------------")
    print(popularity.head(10))

    # category analysis
    merged = order_items.merge(
        products,
        on="product_id",
        how="left",
    )

    category_sales = (
        merged
        .dropna(subset=["product_category_name_english"])
        .groupby("product_category_name_english")
        .size()
        .reset_index(name="sales")
        .sort_values("sales", ascending=False)
    )

    save_result(category_sales, "category_sales.csv")

    print("\nTop Categories")
    print(category_sales.head(10))

    # monthly payment value
    revenue = payments.merge(
        orders,
        on="order_id",
        how="left",
    )

    revenue["order_purchase_timestamp"] = pd.to_datetime(
        revenue["order_purchase_timestamp"],
        errors="coerce",
    )

    revenue["Month"] = (
        revenue["order_purchase_timestamp"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly = (
        revenue
        .dropna(subset=["Month"])
        .groupby("Month")["payment_value"]
        .sum()
        .reset_index(name="payment_value")
    )

    save_result(monthly, "monthly_revenue.csv")

    # payment methods
    payment_stats = payments["payment_type"].value_counts().reset_index()
    payment_stats.columns = ["Payment Method", "Transactions"]

    save_result(payment_stats, "payment_statistics.csv")

    # financial KG application
    financial_kg_profile = derive_financial_kg_profiles(
        payments,
        orders,
        customers,
        order_items,
        products,
    )

    save_result(
        financial_kg_profile,
        "financial_kg_customer_profiles.csv",
    )

    print("\n==============================")
    print(" FINANCIAL KG APPLICATION")
    print("==============================")
    print(
        "Customer financial profiles connect payment value with "
        "order activity and product/category relationships."
    )
    print("\nTop 10 Financial KG Customer Profiles")
    print(
        financial_kg_profile[
            [
                "customer_unique_id",
                "total_payment_value",
                "order_count",
                "average_payment_per_order",
                "distinct_products",
                "distinct_categories",
            ]
        ].head(10)
    )

    # KG business interpretation
    print("\n==============================")
    print(" KG BUSINESS INSIGHTS")
    print("==============================")

    print(
        """
• High-spending customers become analytically important KG entities.
• Frequently purchased products can act as popular entities.
• Repeat customers provide observable behavioural patterns.
• Payment nodes enrich transaction-related analysis.
• Financial profiles connect monetary activity with product relationships.
• Derived results can support downstream ML and analytical services.
"""
    )

    print("\nAll KG analytics saved inside /results folder")

    return {
        "customer_clv": customer_spending,
        "repeat_customers": repeat,
        "customer_identity_validation": identity_validation,
        "product_popularity": popularity,
        "category_sales": category_sales,
        "monthly_revenue": monthly,
        "payment_statistics": payment_stats,
        "financial_kg_profile": financial_kg_profile,
    }