# Multi-Layer Knowledge Graph for E-commerce Customer Behavior and Financial Analysis

## Overview

This project implements a **Knowledge Graph-based analytical system** for e-commerce customer behavior and financial analysis using the **Olist Brazilian E-commerce Dataset**.

The system integrates heterogeneous e-commerce data into a multi-relational Knowledge Graph and provides graph-based analytical services for exploring:

* Customer behavior and purchasing patterns
* Orders, products, categories, and sellers
* Payment and transaction information
* Customer spending and average order value
* Product and category popularity
* Review and sentiment information
* Seller performance
* Geographic relationships and clusters
* Graph embeddings using Node2Vec

The project focuses on using a Knowledge Graph to represent relationships between heterogeneous entities and to provide interpretable graph-based analytical services.

---

## Knowledge Graph Architecture

The system follows a layered architecture in which source data is transformed into a Knowledge Graph and subsequently enriched with derived analytical information.

### 1. Source / Extensional Data Layer

The system loads and preprocesses the Olist datasets, including:

* Customers
* Orders
* Order items
* Products
* Payments
* Sellers
* Reviews
* Geolocation data
* Product category translations

These datasets provide the source facts from which the Knowledge Graph is constructed.

### 2. Knowledge Graph Layer

The main graph is implemented using **NetworkX** as a directed property graph.

Important entity types include:

* Customer
* Order
* Product
* Seller
* Payment
* Category
* Original Category
* Review
* Sentiment
* City
* State
* Geographic Cluster

Examples of relationships include:

```text
Customer ──PLACED──────────> Order
Order ──CONTAINS───────────> Product
Product ──BELONGS_TO───────> Category
Product ──OFFERED_BY───────> Seller
Order ──HAS_PAYMENT────────> Payment
Order ──HAS_REVIEW─────────> Review
Review ──HAS_SENTIMENT─────> Sentiment
Customer ──LOCATED_IN──────> City
City ──LOCATED_IN──────────> State
```

This representation enables multi-hop graph traversal and analytical queries across otherwise separate datasets.

### 3. Derived Knowledge and Analytical Layer

The system derives additional information from the graph and source data, including:

* Repeat-customer identification
* Customer spending and average order value
* Product popularity
* Category popularity
* Seller-level analytical metrics
* Review sentiment categories
* Geographic clusters
* Other graph-based statistics and metrics

These derived results demonstrate how the Knowledge Graph can be enriched with information that is not explicitly stored as a single source fact.

### 4. Representation Learning Layer

The project applies **Node2Vec** to learn vector representations of graph nodes.

Node2Vec is used as a **Knowledge Graph / graph embedding technique** for exploring structural similarity between entities.

The project therefore addresses graph representation learning without claiming implementation of LO3.

### 5. Service and Application Layer

A Streamlit application provides an interactive interface to the Knowledge Graph.

The application supports:

* Knowledge Graph overview
* Natural-language-style analytical queries
* Entity search
* Local graph exploration
* Relationship inspection
* Graph visualization
* Graph metrics
* Customer and seller analytics
* Saved analytical results

The natural-language query interface uses **deterministic intent detection and graph traversal** rather than an external Large Language Model.

---

## Key Features

### Knowledge Graph Construction

* Multi-relational property graph
* Schema-aware entity and relationship types
* Integration of multiple Olist datasets
* Customer, order, product, seller, payment, review, and category relationships
* Category translation representation
* Geographic entity representation

### Knowledge Graph Analytics

* Customer spending analysis
* Repeat-customer identification
* Average order value
* Transaction/payment value analysis
* Product popularity
* Category popularity
* Seller-level analysis
* Review sentiment summaries
* Graph structural metrics

### Knowledge Graph Enrichment

* Derived customer and product analytics
* Review sentiment categorization
* Geographic clustering
* Additional analytical nodes and relationships

The enrichment process demonstrates a basic form of Knowledge Graph evolution by adding derived information to the graph.

### Graph Embeddings

* Node2Vec-based graph embeddings
* Nearest-neighbor exploration
* Structural similarity evaluation
* Embedding statistics
* t-SNE visualization

The embedding evaluation uses a **project-defined structural evaluation**, where embedding neighbors are compared with graph connectivity within a limited hop distance. This is intended as a structural sanity check rather than a standard supervised link-prediction benchmark.

### Knowledge Graph Service

The system provides a deterministic analytical query service capable of answering questions related to:

* Customer spending
* Repeat customers
* Product popularity
* Category popularity
* Seller popularity
* Average order value
* Payment/transaction value
* Customer products
* Customer categories
* Customer sellers
* Review sentiment

The service returns analytical results and, where applicable, graph paths supporting the result.

### Visualization

* Interactive PyVis graph visualization
* Entity search
* Local neighborhood exploration
* Entity-type exploration
* Relationship inspection
* Streamlit-based dashboard

---

## Dataset

The project uses the **Olist Brazilian E-commerce Dataset**.

The dataset contains interconnected information about:

* Customers
* Orders
* Order items
* Products
* Payments
* Sellers
* Reviews
* Geolocation
* Product categories

The heterogeneous structure of the dataset makes it suitable for demonstrating Knowledge Graph construction and multi-hop relationship analysis in an e-commerce setting.

---

## Technologies

* **Python** — implementation
* **NetworkX** — Knowledge Graph construction and traversal
* **Pandas** — data loading and preprocessing
* **NumPy** — numerical processing
* **Scikit-learn** — clustering and analytical processing
* **Node2Vec** — graph embeddings
* **TextBlob** — NLP-based review analysis
* **PyVis** — interactive graph visualization
* **Streamlit** — Knowledge Graph service interface
* **KaggleHub** — dataset acquisition

---

## Learning Outcomes Covered

The project uses the official course learning-outcome definitions.

### Primary Learning Outcomes

**LO7 — Apply a system to create a Knowledge Graph**

The project applies a Python-based pipeline to acquire, preprocess, integrate, and transform the Olist datasets into a multi-relational Knowledge Graph.

**LO9 — Describe real-world applications of Knowledge Graphs**

The project demonstrates a real-world e-commerce application involving customer behavior, purchasing relationships, product/category analysis, sellers, payments, reviews, and geographic information.

### Additional Learning Outcomes

**LO1 — Understand and apply Knowledge Graph Embeddings**

Node2Vec is applied to obtain vector representations of graph entities and evaluate their structural similarity.

**LO2 — Understand and apply logical knowledge in KGs**

The project derives additional knowledge from existing graph facts through rule-based and graph-based reasoning operations.

**LO4 — Compare different Knowledge Graph data models from database, semantic web, machine learning and data science communities**

The project compares relational, RDF/Semantic Web, property graph, and document-oriented representations and explains the choice of a property graph for the implementation.

**LO5 — Design and implement architectures of a Knowledge Graph**

The project implements a layered architecture covering data ingestion, graph construction, enrichment, analytics, representation learning, querying, and visualization.

**LO6 — Describe and apply scalable reasoning methods in Knowledge Graphs**

The project demonstrates rule-based and graph traversal approaches for deriving information from interconnected entities. Scalability limitations of the in-memory NetworkX implementation are acknowledged.

**LO8 — Apply a system to evolve a Knowledge Graph**

The system enriches the graph with derived analytical information such as sentiment categories and geographic clusters.

**LO11 — Apply a system to provide services through a Knowledge Graph**

The Streamlit application provides an interactive service for querying, exploring, visualizing, and analysing the Knowledge Graph.

**LO12 — Describe connections between Knowledge Graphs, Machine Learning (ML), and Artificial Intelligence (AI)**

The project demonstrates connections between Knowledge Graphs, graph embeddings, NLP-based enrichment, analytical processing, and application-level graph services.

### Learning Outcomes Outside the Project Scope

**LO3 — Understand and apply Graph Neural Networks**

Not included. The project uses Node2Vec for graph embeddings but does not implement a Graph Neural Network.

**LO10 — Describe financial Knowledge Graph applications**

Financial and transaction-related analysis is present in the project, but LO10 is not selected as a dedicated learning outcome for the project scope.

---

## Data Model Choice

The project uses a **property graph representation** implemented with NetworkX.

A property graph was selected because the application requires:

* Direct relationship traversal
* Typed relationships
* Entity properties
* Multi-hop exploration
* Graph-based analytics
* Flexible representation of heterogeneous e-commerce entities

The project also includes a comparison with relational, RDF/Semantic Web, and document-oriented approaches.

The comparison is conceptual and is used to justify the representation choice; it is not intended as a performance benchmark between database systems.

---

## Project Scope and Limitations

The project is designed as a manageable 3 ECTS Knowledge Graph implementation rather than a production-scale KGMS.

Important limitations include:

* The graph is held in memory using NetworkX.
* The system is not designed for distributed or very large-scale graph processing.
* Node2Vec is used instead of a Graph Neural Network.
* Embedding evaluation uses a project-defined structural evaluation rather than a labelled link-prediction benchmark.
* Derived analytical metrics are descriptive and should not automatically be interpreted as formal financial models.
* Seller composite scores are project-defined analytical measures.
* Historical customer spending should be distinguished from a predictive customer-lifetime-value model.
* Geographic clusters are derived analytical entities rather than original source facts.
* The natural-language query interface uses deterministic intent detection rather than an LLM.

These limitations are considered when interpreting the results.

---

## How to Run

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Run the main Knowledge Graph pipeline:

```bash
python main.py
```

To launch the Streamlit application:

```bash
streamlit run streamlit_app.py
```

The application uses the generated Knowledge Graph and analytical result files to provide interactive exploration and Knowledge Graph services.
