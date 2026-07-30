from sklearn.cluster import KMeans
import pandas as pd
import os



# ----------------------------------------------------
# GEO-SPATIAL KG ANALYSIS
# CUSTOMER LOCATION SUBGRAPH DISCOVERY
# ----------------------------------------------------

def run_geo_analysis(geo):


    print("\n--- GEOSPATIAL KG SUBGRAPH DISCOVERY ---")



    # ------------------------------------------------
    # Create results folder
    # ------------------------------------------------


    os.makedirs(
        "results",
        exist_ok=True
    )



    # ------------------------------------------------
    # Data Cleaning
    # ------------------------------------------------


    geo_clean = geo.dropna(

        subset=[

            "geolocation_lat",

            "geolocation_lng"

        ]

    ).copy()



    coords = geo_clean[

        [

            "geolocation_lat",

            "geolocation_lng"

        ]

    ]



    print(

        "Geographical points analysed:",

        len(coords)

    )



    # ------------------------------------------------
    # K-Means Clustering
    # ------------------------------------------------


    kmeans = KMeans(

        n_clusters=8,

        n_init=10,

        random_state=42

    )



    geo_clean["cluster"] = kmeans.fit_predict(

        coords

    )



    # ------------------------------------------------
    # Cluster Statistics
    # ------------------------------------------------


    cluster_stats = (

        geo_clean

        .groupby("cluster")

        .agg(

            points=(

                "geolocation_lat",

                "count"

            ),


            avg_latitude=(

                "geolocation_lat",

                "mean"

            ),


            avg_longitude=(

                "geolocation_lng",

                "mean"

            )

        )

        .sort_values(

            "points",

            ascending=False

        )

    )



    print(

        "\nDERIVED FACT: Geographic KG Clusters"

    )


    print(

        cluster_stats

    )



    # ------------------------------------------------
    # Add KG Interpretation Labels
    # ------------------------------------------------


    cluster_stats["kg_meaning"] = (

        "Regional geographic entity cluster"

    )



    # ------------------------------------------------
    # Save Results
    # ------------------------------------------------


    cluster_file = (

        "results/"

        "geographic_clusters.csv"

    )


    geo_file = (

        "results/"

        "geo_points_with_clusters.csv"

    )



    cluster_stats.to_csv(

        cluster_file

    )


    geo_clean.to_csv(

        geo_file,

        index=False

    )



    print(

        "\nSaved geographic cluster results:"

    )


    print(

        cluster_file

    )


    print(

        geo_file

    )



    # ------------------------------------------------
    # Interpretation Layer
    # ------------------------------------------------


    print(

        "\n--- GEO KG INTERPRETATION ---"

    )


    print(

        """
    Geographic clusters represent latent regional
    structures in the ecommerce graph.


    Possible KG interpretation:


    Customer Location
            |
            |
        Regional Cluster
            |
            |
    Demand / Supply Pattern



    These clusters support:

        - regional demand analysis
        - seller coverage analysis
        - logistics optimisation
        - geographic recommendation systems

    In KG terms:

    LOCATION nodes can be enriched with
    cluster membership information.
    """

    )



    return geo_clean