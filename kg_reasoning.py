import os
import time

import pandas as pd


# project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
REASONING_DIR = os.path.join(RESULTS_DIR, "reasoning")

# bounded reasoning safeguards
MAX_ITERATIONS = 5
MAX_TOTAL_NEW_FACTS = 500_000
MAX_NEW_FACTS_PER_RULE = 150_000


# explicit logical rules
RULES = [
    {
        "name": "customer_product",
        "body": [
            ("PLACED", "customer", "order"),
            ("CONTAINS", "order", "product"),
        ],
        "head": "CUSTOMER_INTERACTED_WITH_PRODUCT",
        "description": (
            "PLACED(customer, order) and CONTAINS(order, product) "
            "imply INTERACTED_WITH(customer, product)"
        ),
    },
    {
        "name": "customer_category",
        "body": [
            ("CUSTOMER_INTERACTED_WITH_PRODUCT", "customer", "product"),
            ("BELONGS_TO", "product", "category"),
        ],
        "head": "CUSTOMER_ASSOCIATED_WITH_CATEGORY",
        "description": (
            "INTERACTED_WITH(customer, product) and "
            "BELONGS_TO(product, category) imply "
            "ASSOCIATED_WITH(customer, category)"
        ),
    },
    {
        "name": "customer_seller",
        "body": [
            ("CUSTOMER_INTERACTED_WITH_PRODUCT", "customer", "product"),
            ("OFFERED_BY", "product", "seller"),
        ],
        "head": "CUSTOMER_ASSOCIATED_WITH_SELLER",
        "description": (
            "INTERACTED_WITH(customer, product) and "
            "OFFERED_BY(product, seller) imply "
            "ASSOCIATED_WITH(customer, seller)"
        ),
    },
    {
        "name": "order_sentiment",
        "body": [
            ("HAS_REVIEW", "order", "review"),
            ("HAS_NLP_SENTIMENT", "review", "sentiment"),
        ],
        "head": "ORDER_HAS_INFERRED_NLP_SENTIMENT",
        "description": (
            "HAS_REVIEW(order, review) and "
            "HAS_NLP_SENTIMENT(review, sentiment) imply "
            "ORDER_HAS_INFERRED_NLP_SENTIMENT(order, sentiment)"
        ),
    },
]


# helper functions
def _relation(data):
    return str(data.get("relation", "")).upper()


def _explicit_facts(G):
    """Extract explicit KG edges as logical facts."""
    facts = set()

    for source, target, data in G.edges(data=True):
        relation = _relation(data)

        if relation:
            facts.add((relation, source, target))

    return facts


def _index_facts(facts):
    """
    Build relation indexes for fast logical joins.

    Each relation is stored as:
        relation -> set((source, target))
    """
    relation_index = {}

    for relation, source, target in facts:
        relation_index.setdefault(relation, set()).add(
            (source, target)
        )

    return relation_index


def _build_join_index(facts, position):
    """Index binary facts by source or target."""
    index = {}

    for source, target in facts:
        key = source if position == 0 else target
        index.setdefault(key, []).append((source, target))

    return index


def _apply_rule(
    relation_index,
    rule,
    existing_facts,
    max_rule_facts=MAX_NEW_FACTS_PER_RULE,
):
    """
    Apply one two-condition rule using indexed joins.

    The implementation avoids Cartesian products and avoids
    constructing a large intermediate join table.
    """
    first_relation, first_left, first_right = rule["body"][0]
    second_relation, second_left, second_right = rule["body"][1]

    first_facts = relation_index.get(first_relation, set())
    second_facts = relation_index.get(second_relation, set())

    if not first_facts or not second_facts:
        return set()

    shared_variables = (
        set((first_left, first_right))
        & set((second_left, second_right))
    )

    if len(shared_variables) != 1:
        return set()

    shared_variable = next(iter(shared_variables))

    first_shared_position = (
        0 if first_left == shared_variable else 1
    )
    second_shared_position = (
        0 if second_left == shared_variable else 1
    )

    second_index = _build_join_index(
        second_facts,
        second_shared_position,
    )

    derived = set()
    head = rule["head"]

    for first_source, first_target in first_facts:
        join_key = (
            first_source
            if first_shared_position == 0
            else first_target
        )

        matches = second_index.get(join_key, [])

        for second_source, second_target in matches:
            first_bindings = {
                first_left: first_source,
                first_right: first_target,
            }

            second_bindings = {
                second_left: second_source,
                second_right: second_target,
            }

            bindings = {
                **first_bindings,
                **second_bindings,
            }

            if head == "CUSTOMER_INTERACTED_WITH_PRODUCT":
                fact = (
                    head,
                    bindings["customer"],
                    bindings["product"],
                )

            elif head == "CUSTOMER_ASSOCIATED_WITH_CATEGORY":
                fact = (
                    head,
                    bindings["customer"],
                    bindings["category"],
                )

            elif head == "CUSTOMER_ASSOCIATED_WITH_SELLER":
                fact = (
                    head,
                    bindings["customer"],
                    bindings["seller"],
                )

            elif head == "ORDER_HAS_INFERRED_NLP_SENTIMENT":
                fact = (
                    head,
                    bindings["order"],
                    bindings["sentiment"],
                )

            else:
                continue

            if fact in existing_facts or fact in derived:
                continue

            derived.add(fact)

            if len(derived) >= max_rule_facts:
                return derived

    return derived


def forward_chain(
    initial_facts,
    rules,
    max_iterations=MAX_ITERATIONS,
    max_total_new_facts=MAX_TOTAL_NEW_FACTS,
    max_new_facts_per_rule=MAX_NEW_FACTS_PER_RULE,
):
    """
    Apply bounded forward chaining until a fixed point is reached.

    Indexed joins avoid Cartesian products. The engine continues
    through all rules instead of allowing one large rule to prevent
    other logical consequences from being derived.
    """
    all_facts = set(initial_facts)
    derivations = []

    for iteration in range(1, max_iterations + 1):
        relation_index = _index_facts(all_facts)
        new_facts = set()

        print(
            f"\nLogical iteration {iteration}: "
            f"{len(all_facts):,} known facts"
        )

        for rule in rules:
            remaining_capacity = (
                max_total_new_facts - len(new_facts)
            )

            if remaining_capacity <= 0:
                print(
                    "  maximum total derived fact limit reached."
                )
                break

            rule_limit = min(
                max_new_facts_per_rule,
                remaining_capacity,
            )

            candidates = _apply_rule(
                relation_index,
                rule,
                all_facts,
                max_rule_facts=rule_limit,
            )

            rule_new = 0

            for fact in candidates:
                if fact in all_facts or fact in new_facts:
                    continue

                if len(new_facts) >= max_total_new_facts:
                    break

                new_facts.add(fact)
                rule_new += 1

                derivations.append(
                    {
                        "iteration": iteration,
                        "rule": rule["name"],
                        "relation": fact[0],
                        "source": fact[1],
                        "target": fact[2],
                        "rule_description": rule["description"],
                    }
                )

            print(
                f"  {rule['name']}: "
                f"{rule_new:,} new facts"
            )

            if rule_new >= rule_limit:
                print(
                    f"  rule safety limit reached: "
                    f"{rule_limit:,}"
                )

        if not new_facts:
            print(
                "No new facts derived. "
                "Logical fixed point reached."
            )
            break

        all_facts.update(new_facts)

        print(
            f"Iteration {iteration} total new facts: "
            f"{len(new_facts):,}"
        )

        if len(new_facts) >= max_total_new_facts:
            print(
                "Maximum total derived fact limit reached."
            )
            break

    else:
        print(
            f"Maximum reasoning iterations reached: "
            f"{max_iterations}"
        )

    return all_facts, pd.DataFrame(derivations)


def _derived_facts_by_relation(
    derivations,
    relation,
):
    if derivations.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "target",
                "inferred_relation",
                "rule",
                "iteration",
            ]
        )

    result = derivations[
        derivations["relation"] == relation
    ].copy()

    return (
        result[
            [
                "source",
                "target",
                "relation",
                "rule",
                "iteration",
            ]
        ]
        .rename(
            columns={
                "relation": "inferred_relation",
            }
        )
        .drop_duplicates()
    )


def create_reasoning_summary(
    initial_facts,
    final_facts,
    derivations,
):
    os.makedirs(
        REASONING_DIR,
        exist_ok=True,
    )

    summary_rows = [
        {
            "metric": "explicit_facts",
            "value": len(initial_facts),
        },
        {
            "metric": "total_facts_after_reasoning",
            "value": len(final_facts),
        },
        {
            "metric": "new_inferred_facts",
            "value": len(derivations),
        },
        {
            "metric": "rules_used",
            "value": (
                derivations["rule"].nunique()
                if not derivations.empty
                else 0
            ),
        },
        {
            "metric": "maximum_inference_depth",
            "value": (
                derivations["iteration"].max()
                if not derivations.empty
                else 0
            ),
        },
        {
            "metric": "max_total_new_facts_limit",
            "value": MAX_TOTAL_NEW_FACTS,
        },
        {
            "metric": "max_new_facts_per_rule",
            "value": MAX_NEW_FACTS_PER_RULE,
        },
    ]

    summary = pd.DataFrame(summary_rows)

    path = os.path.join(
        REASONING_DIR,
        "reasoning_summary.csv",
    )

    summary.to_csv(
        path,
        index=False,
    )

    print(
        f"\nReasoning summary saved:\n{path}"
    )

    return summary


def save_reasoning_results(derivations):
    os.makedirs(
        REASONING_DIR,
        exist_ok=True,
    )

    if derivations.empty:
        print(
            "\nNo new logical facts were inferred."
        )
        return

    derivations_path = os.path.join(
        REASONING_DIR,
        "logical_derivations.csv",
    )

    derivations.to_csv(
        derivations_path,
        index=False,
    )

    print(
        f"\nLogical derivations saved:\n"
        f"{derivations_path}"
    )

    relation_counts = (
        derivations["relation"]
        .value_counts()
        .reset_index()
    )

    relation_counts.columns = [
        "inferred_relation",
        "fact_count",
    ]

    relation_path = os.path.join(
        REASONING_DIR,
        "inferred_relation_counts.csv",
    )

    relation_counts.to_csv(
        relation_path,
        index=False,
    )

    print(
        f"Inferred relation counts saved:\n"
        f"{relation_path}"
    )


def add_inferred_facts_to_graph(
    G,
    derivations,
):
    """
    Evolve the KG by adding logically inferred edges.

    Derived edges are explicitly marked as inferred so they
    remain distinguishable from source-dataset relationships.
    """
    if derivations.empty:
        return 0

    added = 0

    unique_derivations = derivations.drop_duplicates(
        subset=["relation", "source", "target"]
    )

    for _, row in unique_derivations.iterrows():
        source = row["source"]
        target = row["target"]
        relation = row["relation"]

        if G.has_edge(source, target):
            continue

        G.add_edge(
            source,
            target,
            relation=relation,
            source="logical_forward_chaining",
            inferred=True,
            inference_rule=row["rule"],
            inference_iteration=int(
                row["iteration"]
            ),
        )

        added += 1

    return added


# main logical reasoning engine
def run_kg_reasoning(G):
    print("\n==============================")
    print(" KNOWLEDGE GRAPH LOGICAL REASONING")
    print("==============================")
    print("\nLO2 — Logical rule-based inference")
    print("LO6 — Bounded forward-chaining reasoning")

    if G is None:
        print(
            "\nReasoning skipped: graph is None."
        )
        return {}

    start_time = time.perf_counter()

    try:
        # extract explicit knowledge
        initial_facts = _explicit_facts(G)

        print(
            f"\nExplicit KG facts: "
            f"{len(initial_facts):,}"
        )

        # apply indexed bounded forward chaining
        final_facts, derivations = forward_chain(
            initial_facts,
            RULES,
            max_iterations=MAX_ITERATIONS,
            max_total_new_facts=MAX_TOTAL_NEW_FACTS,
            max_new_facts_per_rule=MAX_NEW_FACTS_PER_RULE,
        )

        print(
            f"\nTotal KG facts after reasoning: "
            f"{len(final_facts):,}"
        )

        print(
            f"New inferred facts: "
            f"{len(derivations):,}"
        )

        # save logical derivations
        save_reasoning_results(derivations)

        # expose individual logical consequences
        results = {
            "customer_product": (
                _derived_facts_by_relation(
                    derivations,
                    "CUSTOMER_INTERACTED_WITH_PRODUCT",
                )
            ),
            "customer_category": (
                _derived_facts_by_relation(
                    derivations,
                    "CUSTOMER_ASSOCIATED_WITH_CATEGORY",
                )
            ),
            "customer_seller": (
                _derived_facts_by_relation(
                    derivations,
                    "CUSTOMER_ASSOCIATED_WITH_SELLER",
                )
            ),
            "order_sentiment": (
                _derived_facts_by_relation(
                    derivations,
                    "ORDER_HAS_INFERRED_NLP_SENTIMENT",
                )
            ),
            "logical_derivations": derivations,
        }

        # evolve the KG with inferred knowledge
        added_edges = add_inferred_facts_to_graph(
            G,
            derivations,
        )

        print(
            f"\nInferred KG edges added: "
            f"{added_edges:,}"
        )

        create_reasoning_summary(
            initial_facts,
            final_facts,
            derivations,
        )

        execution_time = (
            time.perf_counter() - start_time
        )

        print(
            f"\nReasoning execution time: "
            f"{execution_time:.2f} seconds"
        )

        print("\n==============================")
        print(
            "LOGICAL REASONING COMPLETED"
        )
        print("==============================")

        return results

    except Exception as error:
        print(
            "\nLogical reasoning execution failed:"
        )
        print(error)
        return {}


# main
if __name__ == "__main__":
    print(
        "Run kg_reasoning.py through main.py "
        "with the constructed NetworkX graph."
    )