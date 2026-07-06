# Multi-Layer Knowledge Graph for E-commerce Customer Behavior and Financial Analysis (KGMS-Based System)

## Overview
This project builds a **multi-layer Knowledge Graph Management System (KGMS)** using the Olist Brazilian E-commerce dataset.

The system integrates:
- Knowledge Graph construction
- Rule-based reasoning (Datalog-inspired)
- Graph embeddings (Node2Vec as GNN approximation)
- NLP sentiment enrichment
- Geospatial clustering
- Financial analytics

The goal is to demonstrate how **Knowledge Graphs unify structured + unstructured + relational data into a single reasoning system**.

---

## KGMS Architecture

The system follows a classical KGMS architecture:

### 1. Ground Extensional Layer
- Raw datasets (customers, orders, products, payments)

### 2. Intensional Layer (Reasoning)
- Derived facts:
  - Customer Lifetime Value
  - Repeat customers
  - Product popularity
  - Category trends

### 3. Subsymbolic Layer
- Node2Vec embeddings
- Interpreted as a simplified GNN message passing process

### 4. External Data Layer
- NLP sentiment analysis (reviews)
- Geospatial clustering

---

## Key Features

- Multi-relational Knowledge Graph construction
- Schema-aware graph modeling
- Rule-based reasoning (Datalog-style inference)
- Customer behavior analysis
- Financial analytics (revenue, CLV)
- Seller performance analysis
- NLP sentiment enrichment
- Geospatial clustering
- Node2Vec embeddings (GNN-equivalent reasoning)
- Graph visualization (PyVis)
- JSON export for interoperability

---

## Dataset

Olist Brazilian E-commerce Dataset (Kaggle)

Includes:
- Customers
- Orders
- Products
- Payments
- Sellers
- Reviews
- Geolocation data
- Product categories

---

## Technologies

- Python
- NetworkX (Graph modeling)
- Pandas (data processing)
- Scikit-learn (clustering)
- Node2Vec (graph embeddings)
- TextBlob (NLP sentiment)
- PyVis (graph visualization)

---

## Learning Outcomes Covered

### Core (Strong proficiency)
- LO7: Knowledge representation using graphs
- LO8: Data integration from heterogeneous sources
- LO9: Machine learning on graph structures

### Additional (Basic proficiency)
- LO1, LO2: Data preprocessing & ingestion
- LO4: Data analytics
- LO5: Graph modeling
- LO6: Data interpretation
- LO11: Visualization
- LO12: Applied AI pipeline design

---

## How to Run

```bash
pip install -r requirements.txt
python main.py