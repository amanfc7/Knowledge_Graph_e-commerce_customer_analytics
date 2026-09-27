import os
import pandas as pd

# result storage
RESULT_FOLDER = "results"


def save_result(df, filename):
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    path = os.path.join(RESULT_FOLDER, filename)
    df.to_csv(path, index=False)
    print("Saved result:", path)


# seller analysis
def run_seller_analysis(order_items, payments):
    """
    Analyse seller performance using order-item-level values.

    Seller sales are calculated from order_items instead of joining
    payments directly to order items. This avoids duplicating payment
    values when an order contains multiple items or payment records.

    The resulting sales value is based on item prices and does not
    represent the total payment value of the order.
    """
    print("\n==============================")
    print(" SELLER ECONOMIC ANALYSIS")
    print("==============================")

    os.makedirs(RESULT_FOLDER, exist_ok=True)

    # seller sales and activity
    seller_revenue = (
        order_items
        .groupby("seller_id", as_index=False)
        .agg(
            sales_value=("price", "sum"),
            freight_value=("freight_value", "sum"),
            transaction_count=("order_id", "nunique"),
            item_count=("order_item_id", "count"),
            product_count=("product_id", "nunique"),
        )
    )

    seller_revenue["total_item_value"] = (
        seller_revenue["sales_value"]
        + seller_revenue["freight_value"]
    )

    seller_revenue["average_order_value"] = (
        seller_revenue["sales_value"]
        / seller_revenue["transaction_count"].replace(0, pd.NA)
    )

    # product diversity is represented by the number of distinct products sold
    def normalize(series):
        min_value = series.min()
        max_value = series.max()

        if max_value == min_value:
            return pd.Series(1.0, index=series.index)

        return (series - min_value) / (max_value - min_value)

    # project-defined composite business score
    seller_revenue["sales_value_score"] = normalize(
        seller_revenue["sales_value"]
    )
    seller_revenue["transaction_score"] = normalize(
        seller_revenue["transaction_count"]
    )
    seller_revenue["product_diversity_score"] = normalize(
        seller_revenue["product_count"]
    )
    seller_revenue["aov_score"] = normalize(
        seller_revenue["average_order_value"]
    )

    seller_revenue["business_score"] = (
        seller_revenue["sales_value_score"]
        + seller_revenue["transaction_score"]
        + seller_revenue["product_diversity_score"]
        + seller_revenue["aov_score"]
    ) / 4

    seller_revenue = seller_revenue.sort_values(
        "business_score",
        ascending=False
    )

    seller_revenue["business_rank"] = range(
        1, len(seller_revenue) + 1
    )

    save_result(
        seller_revenue,
        "seller_performance_analysis.csv"
    )

    # top sellers
    print("\n------------------------------")
    print("Top Sellers by Product Sales")
    print("------------------------------")

    print(
        seller_revenue[
            [
                "seller_id",
                "sales_value",
                "transaction_count",
                "product_count",
                "average_order_value",
            ]
        ].head(10)
    )

    # seller analysis summary
    print("\n------------------------------")
    print("Seller Analysis Summary")
    print("------------------------------")

    print(
        f"Number of sellers analysed: "
        f"{seller_revenue['seller_id'].nunique():,}"
    )

    print(
        f"Total item sales value: "
        f"{seller_revenue['sales_value'].sum():,.2f}"
    )

    print(
        f"Total freight value: "
        f"{seller_revenue['freight_value'].sum():,.2f}"
    )

    print(
        "\nSeller sales are based on order-item prices. "
        "Payment values are not directly joined to order items "
        "to avoid duplicate aggregation."
    )

    return seller_revenue