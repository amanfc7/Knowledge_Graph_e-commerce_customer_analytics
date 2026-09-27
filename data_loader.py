
import os
import kagglehub
import pandas as pd


# configuration

DATASET_NAME = "olistbr/brazilian-ecommerce"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
VALIDATION_DIR = os.path.join(RESULTS_DIR, "validation")


# save data quality report

def save_data_quality_report(datasets, filename="data_quality_report.csv"):
    os.makedirs(VALIDATION_DIR, exist_ok=True)

    rows = []
    for name, df in datasets.items():
        rows.append({
            "dataset": name,
            "rows": len(df),
            "columns": len(df.columns),
            "duplicate_rows": int(df.duplicated().sum()),
            "missing_values": int(df.isna().sum().sum())
        })

    path = os.path.join(VALIDATION_DIR, filename)
    pd.DataFrame(rows).to_csv(path, index=False)

    print("\nData quality report saved to:", path)


# validate required columns

def validate_columns(datasets, required_columns):
    for name, columns in required_columns.items():
        missing = set(columns) - set(datasets[name].columns)
        if missing:
            raise ValueError(
                f"{name} is missing required columns: {sorted(missing)}"
            )


# clean string columns

def clean_string_columns(df, columns):
    for column in columns:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()
    return df


# load data

def load_data():
    print("\n--- loading KG data layer (ground facts) ---")

    # dataset download

    path = kagglehub.dataset_download(DATASET_NAME)

    print("Dataset path:", path)
    print("Files found:", os.listdir(path))

    # core KG entities

    customers = pd.read_csv(
        os.path.join(path, "olist_customers_dataset.csv"),
        encoding="utf-8"
    )
    orders = pd.read_csv(
        os.path.join(path, "olist_orders_dataset.csv"),
        encoding="utf-8"
    )
    order_items = pd.read_csv(
        os.path.join(path, "olist_order_items_dataset.csv"),
        encoding="utf-8"
    )
    products = pd.read_csv(
        os.path.join(path, "olist_products_dataset.csv"),
        encoding="utf-8"
    )
    payments = pd.read_csv(
        os.path.join(path, "olist_order_payments_dataset.csv"),
        encoding="utf-8"
    )

    # extension knowledge sources

    sellers = pd.read_csv(
        os.path.join(path, "olist_sellers_dataset.csv"),
        encoding="utf-8"
    )
    reviews = pd.read_csv(
        os.path.join(path, "olist_order_reviews_dataset.csv"),
        encoding="utf-8"
    )
    geo = pd.read_csv(
        os.path.join(path, "olist_geolocation_dataset.csv"),
        encoding="utf-8"
    )

    # category translation knowledge source

    category = pd.read_csv(
        os.path.join(path, "product_category_name_translation.csv"),
        encoding="utf-8"
    )

    datasets = {
        "customers": customers,
        "orders": orders,
        "order_items": order_items,
        "products": products,
        "payments": payments,
        "sellers": sellers,
        "reviews": reviews,
        "geo": geo,
        "category_translation": category
    }

    # validate expected schema

    required_columns = {
        "customers": [
            "customer_id", "customer_unique_id",
            "customer_city", "customer_state"
        ],
        "orders": [
            "order_id", "customer_id",
            "order_status", "order_purchase_timestamp"
        ],
        "order_items": [
            "order_id", "product_id", "seller_id",
            "order_item_id", "price", "freight_value"
        ],
        "products": ["product_id", "product_category_name"],
        "payments": ["order_id", "payment_value"],
        "sellers": [
            "seller_id", "seller_city", "seller_state"
        ],
        "reviews": [
            "review_id", "order_id", "review_score"
        ],
        "geo": [
            "geolocation_zip_code_prefix",
            "geolocation_lat",
            "geolocation_lng"
        ],
        "category_translation": [
            "product_category_name",
            "product_category_name_english"
        ]
    }

    validate_columns(datasets, required_columns)

    # data quality checks

    print("\n--- data validation ---")
    for name, df in datasets.items():
        print(
            f"{name}: rows={len(df)}, "
            f"columns={len(df.columns)}"
        )

    print("\n--- duplicate check ---")
    for name, df in datasets.items():
        print(f"{name}: duplicates={df.duplicated().sum()}")

    print("\n--- missing value check ---")
    for name, df in datasets.items():
        print(f"{name}: missing values={df.isna().sum().sum()}")

    # save raw data quality report before cleaning

    save_data_quality_report(
        datasets,
        "data_quality_report_raw.csv"
    )

    # remove exact duplicate rows

    for name in datasets:
        before = len(datasets[name])
        datasets[name] = datasets[name].drop_duplicates().copy()
        removed = before - len(datasets[name])

        if removed:
            print(f"{name}: removed {removed} duplicate rows")

    (
        customers,
        orders,
        order_items,
        products,
        payments,
        sellers,
        reviews,
        geo,
        category
    ) = (
        datasets["customers"],
        datasets["orders"],
        datasets["order_items"],
        datasets["products"],
        datasets["payments"],
        datasets["sellers"],
        datasets["reviews"],
        datasets["geo"],
        datasets["category_translation"]
    )

    # clean identifiers and descriptive strings

    string_columns = {
        "customers": [
            "customer_id", "customer_unique_id",
            "customer_city", "customer_state"
        ],
        "orders": ["order_id", "customer_id", "order_status"],
        "order_items": ["order_id", "product_id", "seller_id"],
        "products": ["product_id", "product_category_name"],
        "payments": ["order_id", "payment_type"],
        "sellers": ["seller_id", "seller_city", "seller_state"],
        "reviews": ["review_id", "order_id"],
        "category": [
            "product_category_name",
            "product_category_name_english"
        ]
    }

    for name, columns in string_columns.items():
        if name == "category":
            category = clean_string_columns(category, columns)
        else:
            datasets[name] = clean_string_columns(
                datasets[name],
                columns
            )

    # clean required KG identifiers

    customers = customers.dropna(
        subset=["customer_id", "customer_unique_id"]
    )
    orders = orders.dropna(
        subset=["order_id", "customer_id"]
    )
    order_items = order_items.dropna(
        subset=["order_id", "product_id", "seller_id"]
    )
    payments = payments.dropna(
        subset=["order_id", "payment_value"]
    )
    products = products.dropna(
        subset=["product_id"]
    )
    sellers = sellers.dropna(
        subset=["seller_id"]
    )
    reviews = reviews.dropna(
        subset=["review_id", "order_id", "review_score"]
    )

    # temporal feature preparation

    date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]

    for column in date_columns:
        if column in orders.columns:
            orders[column] = pd.to_datetime(
                orders[column],
                errors="coerce"
            )

    # numeric feature preparation

    numeric_columns = {
        "order_items": [
            "order_item_id", "price", "freight_value"
        ],
        "payments": [
            "payment_installments", "payment_value"
        ],
        "reviews": ["review_score"],
        "products": [
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm"
        ],
        "geo": [
            "geolocation_zip_code_prefix",
            "geolocation_lat",
            "geolocation_lng"
        ]
    }

    for name, columns in numeric_columns.items():
        df = datasets[name] if name in datasets else geo

        for column in columns:
            if column in df.columns:
                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )

        if name == "geo":
            geo = df
        else:
            datasets[name] = df

    order_items = datasets["order_items"]
    payments = datasets["payments"]
    products = datasets["products"]
    reviews = datasets["reviews"]

    # category translation knowledge layer

    category = category.dropna(
        subset=[
            "product_category_name",
            "product_category_name_english"
        ]
    ).drop_duplicates(
        subset=["product_category_name"]
    )

    products = products.merge(
        category,
        on="product_category_name",
        how="left",
        validate="many_to_one"
    )

    # geo normalization

    geo = geo.dropna(
        subset=[
            "geolocation_zip_code_prefix",
            "geolocation_lat",
            "geolocation_lng"
        ]
    ).copy()

    # save final data quality report after cleaning

    final_datasets = {
        "customers": customers,
        "orders": orders,
        "order_items": order_items,
        "products": products,
        "payments": payments,
        "sellers": sellers,
        "reviews": reviews,
        "geo": geo,
        "category_translation": category
    }

    save_data_quality_report(
        final_datasets,
        "data_quality_report_cleaned.csv"
    )

    # final data summary

    print("\n--- final data ready for KG ---")
    for name, df in final_datasets.items():
        print(f"{name.replace('_', ' ').title()}: {len(df)}")

    return (
        customers,
        orders,
        order_items,
        products,
        payments,
        sellers,
        reviews,
        geo,
        category
    )
