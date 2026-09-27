import os
import re

import pandas as pd
from sklearn.cluster import KMeans

# geo-spatial KG analysis, customer location subgraph discovery, and KG enrichment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def normalize_city(value):
    """Normalize city names to match city identifiers in the KG."""
    if pd.isna(value):
        return None

    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def run_geo_analysis(geo, G=None):
    """
    Perform geographic clustering and optionally enrich the KG.

    Derived relationships use:
        City -> BELONGS_TO_CLUSTER -> Geographic Cluster

    Geographic clusters are derived analytical entities, not
    original facts from the source dataset.
    """
    print("\n--- GEOSPATIAL KG SUBGRAPH DISCOVERY ---")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # clean geographic coordinates
    geo_clean = geo.dropna(
        subset=["geolocation_lat", "geolocation_lng"]
    ).copy()

    coords = geo_clean[
        ["geolocation_lat", "geolocation_lng"]
    ]

    print("Geographical points analysed:", len(coords))

    if coords.empty:
        print("No valid geographical coordinates found.")
        return geo_clean

    # k-means clustering
    print("\nRunning geographic K-Means clustering...")

    n_clusters = min(8, len(coords))

    if n_clusters < 2:
        geo_clean["cluster"] = 0
    else:
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=42,
        )
        geo_clean["cluster"] = kmeans.fit_predict(coords)

    # cluster statistics
    cluster_stats = (
        geo_clean
        .groupby("cluster")
        .agg(
            points=("geolocation_lat", "count"),
            avg_latitude=("geolocation_lat", "mean"),
            avg_longitude=("geolocation_lng", "mean"),
        )
        .sort_values("points", ascending=False)
    )

    cluster_stats["kg_meaning"] = "Derived regional geographic cluster"

    print("\nDERIVED FACT: Geographic KG Clusters")
    print(cluster_stats)

    # normalize city names
    if "geolocation_city" in geo_clean.columns:
        geo_clean["city_normalized"] = (
            geo_clean["geolocation_city"].apply(normalize_city)
        )
    else:
        geo_clean["city_normalized"] = None

    # determine dominant cluster per city
    city_cluster = (
        geo_clean
        .dropna(subset=["city_normalized"])
        .groupby(["city_normalized", "cluster"])
        .size()
        .reset_index(name="point_count")
    )

    if not city_cluster.empty:
        city_cluster = (
            city_cluster
            .sort_values(
                ["city_normalized", "point_count"],
                ascending=[True, False],
            )
            .drop_duplicates(subset=["city_normalized"])
        )

    # KG enrichment
    kg_edges_added = 0
    kg_nodes_added = 0

    if G is not None and not city_cluster.empty:
        print("\n--- ENRICHING KNOWLEDGE GRAPH WITH GEOGRAPHIC CLUSTERS ---")

        # create geographic cluster nodes
        for cluster_id in sorted(city_cluster["cluster"].unique()):
            cluster_id = int(cluster_id)
            cluster_node = f"GEOGRAPHIC_CLUSTER_{cluster_id}"

            if cluster_node not in G:
                G.add_node(
                    cluster_node,
                    type="geographic_cluster",
                    schema_type="concept",
                    display_name=f"Geographic Cluster {cluster_id}",
                    source="Olist_geolocation_KMeans",
                    cluster_id=cluster_id,
                )
                kg_nodes_added += 1

        # connect existing city nodes
        for _, row in city_cluster.iterrows():
            city = row["city_normalized"]
            cluster_id = int(row["cluster"])

            if not city:
                continue

            cluster_node = f"GEOGRAPHIC_CLUSTER_{cluster_id}"
            city_node = f"CITY_{city}"

            if city_node in G and not G.has_edge(city_node, cluster_node):
                G.add_edge(
                    city_node,
                    cluster_node,
                    relation="BELONGS_TO_CLUSTER",
                    weight=1.0,
                    source="KMeans_Geographic_Analysis",
                )
                kg_edges_added += 1

        print("Geographic cluster nodes added:", kg_nodes_added)
        print("City-cluster relationships added:", kg_edges_added)

    elif G is None:
        print("\nKG enrichment skipped: no NetworkX graph was provided.")

    # save cluster results
    cluster_file = os.path.join(
        RESULTS_DIR, "geographic_clusters.csv"
    )
    geo_file = os.path.join(
        RESULTS_DIR, "geo_points_with_clusters.csv"
    )
    city_cluster_file = os.path.join(
        RESULTS_DIR, "city_geographic_clusters.csv"
    )

    cluster_stats.to_csv(cluster_file)
    geo_clean.to_csv(geo_file, index=False)
    city_cluster.to_csv(city_cluster_file, index=False)

    print("\nSaved geographic cluster results:")
    print(cluster_file)
    print(geo_file)
    print(city_cluster_file)

    # interpretation
    print("\n--- GEO KG INTERPRETATION ---")
    print(
        """
Geographic clusters are derived analytical entities created from
latitude and longitude information.

These derived clusters can support:
    - regional demand analysis
    - customer distribution analysis
    - seller coverage analysis
    - logistics analysis
    - geographic recommendation scenarios

"""
    )

    return geo_clean