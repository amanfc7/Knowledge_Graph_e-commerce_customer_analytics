import kagglehub
import pandas as pd
import os



def load_data():

    print("\n--- LOADING KG DATA LAYER (GROUND FACTS) ---")



    # =====================================================
    # DATASET DOWNLOAD
    # =====================================================

    path = kagglehub.dataset_download(
        "olistbr/brazilian-ecommerce"
    )


    print(
        "Dataset path:",
        path
    )


    print(
        "Files found:",
        os.listdir(path)
    )



    # =====================================================
    # CORE KG ENTITIES
    # =====================================================


    customers = pd.read_csv(
        f"{path}/olist_customers_dataset.csv",
        encoding="utf-8"
    )


    orders = pd.read_csv(
        f"{path}/olist_orders_dataset.csv",
        encoding="utf-8"
    )


    order_items = pd.read_csv(
        f"{path}/olist_order_items_dataset.csv",
        encoding="utf-8"
    )


    products = pd.read_csv(
        f"{path}/olist_products_dataset.csv",
        encoding="utf-8"
    )


    payments = pd.read_csv(
        f"{path}/olist_order_payments_dataset.csv",
        encoding="utf-8"
    )



    # =====================================================
    # EXTENSION KNOWLEDGE SOURCES
    # =====================================================


    sellers = pd.read_csv(
        f"{path}/olist_sellers_dataset.csv",
        encoding="utf-8"
    )


    reviews = pd.read_csv(
        f"{path}/olist_order_reviews_dataset.csv",
        encoding="utf-8"
    )


    geo = pd.read_csv(
        f"{path}/olist_geolocation_dataset.csv",
        encoding="utf-8"
    )



    # IMPORTANT:
    # Category translation knowledge source

    category = pd.read_csv(
        f"{path}/product_category_name_translation.csv",
        encoding="utf-8"
    )



    # =====================================================
    # DATA QUALITY CHECKS
    # LO1 + LO2
    # =====================================================


    print("\n--- DATA VALIDATION ---")



    datasets = {

        "customers":
        customers,

        "orders":
        orders,

        "order_items":
        order_items,

        "products":
        products,

        "payments":
        payments,

        "sellers":
        sellers,

        "reviews":
        reviews,

        "geo":
        geo,

        "category_translation":
        category

    }



    for name, df in datasets.items():

        print(
            f"{name}: rows={len(df)}, columns={len(df.columns)}"
        )



    print("\n--- DUPLICATE CHECK ---")



    for name, df in datasets.items():

        duplicates = df.duplicated().sum()

        print(
            f"{name}: duplicates={duplicates}"
        )



    print("\n--- MISSING VALUE CHECK ---")



    for name, df in datasets.items():

        missing = df.isna().sum().sum()

        print(
            f"{name}: missing values={missing}"
        )



    # =====================================================
    # CLEANING + NORMALIZATION
    # =====================================================



    customers = customers.dropna(
        subset=[
            "customer_id"
        ]
    )



    orders = orders.dropna(
        subset=[
            "order_id",
            "customer_id"
        ]
    )



    order_items = order_items.dropna(
        subset=[
            "order_id",
            "product_id",
            "seller_id"
        ]
    )



    payments = payments.dropna(
        subset=[
            "order_id",
            "payment_value"
        ]
    )



    products = products.dropna(
        subset=[
            "product_id"
        ]
    )



    sellers = sellers.dropna(
        subset=[
            "seller_id"
        ]
    )



    reviews = reviews.dropna(
        subset=[
            "review_id",
            "order_id"
        ]
    )



    # =====================================================
    # TEMPORAL FEATURE PREPARATION
    # Useful for future KG temporal reasoning
    # =====================================================



    date_columns = [

        "order_purchase_timestamp",

        "order_approved_at",

        "order_delivered_carrier_date",

        "order_delivered_customer_date",

        "order_estimated_delivery_date"

    ]



    for col in date_columns:


        if col in orders.columns:


            orders[col] = pd.to_datetime(

                orders[col],

                errors="coerce"

            )



    # =====================================================
    # CATEGORY TRANSLATION KNOWLEDGE LAYER
    # =====================================================


    category = category.dropna(

        subset=[

            "product_category_name",

            "product_category_name_english"

        ]

    )



    products = products.merge(

        category,

        on="product_category_name",

        how="left"

    )



    # =====================================================
    # GEO NORMALIZATION
    # =====================================================


    geo = geo.dropna(

        subset=[

            "geolocation_zip_code_prefix",

            "geolocation_lat",

            "geolocation_lng"

        ]

    )



    # =====================================================
    # FINAL DATA SUMMARY
    # =====================================================


    print("\n--- FINAL DATA READY FOR KG ---")



    print(
        "Customers:",
        len(customers)
    )


    print(
        "Orders:",
        len(orders)
    )


    print(
        "Products:",
        len(products)
    )


    print(
        "Payments:",
        len(payments)
    )


    print(
        "Sellers:",
        len(sellers)
    )


    print(
        "Reviews:",
        len(reviews)
    )


    print(
        "Category translations:",
        len(category)
    )



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