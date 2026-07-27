from sklearn.cluster import KMeans
import pandas as pd



# ----------------------------------------------------
# GEO-SPATIAL KG ANALYSIS
# CUSTOMER LOCATION SUBGRAPH DISCOVERY
# ----------------------------------------------------

def run_geo_analysis(geo):

    print("\n--- GEOSPATIAL KG SUBGRAPH DISCOVERY ---")


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
    # Interpretation Layer
    # ------------------------------------------------

    print(
        "\n--- GEO KG INTERPRETATION ---"
    )


    print(
        """
    Geographic clusters represent latent regional
    structures in the e-commerce graph.

    Possible KG interpretation:

    Customer location
            |
            |
        Regional cluster
            |
            |
    Demand / supply patterns


    These clusters can support:
        - regional demand analysis
        - seller coverage analysis
        - logistics optimisation
    """
    )


    return geo_clean