"""
KAGGLE NOTEBOOK 4: Cross-Partition Cost Estimator (LightGBM)
Target: Free CPU or GPU on Kaggle
Dependencies: pip install lightgbm scikit-learn pandas joblib

Objective:
Trains an interpretable Gradient-Boosted Decision Tree (GBDT) regressor that predicts
cross-partition network cost from AST structural features (hop count, distinct partitions, edge-cuts).
Used for fast test-time candidate re-ranking (Path B candidate scoring).
"""

import json
import os
import re
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def extract_ast_features(record: dict) -> dict:
    cypher = record["gold_cypher"]
    cypher_upper = cypher.upper()

    # 1. Traversal hop count
    hops = cypher.count("-[") + cypher.count("<-[")

    # 2. Partitions touched
    partitions = record.get("referenced_partitions", [0])
    num_distinct_partitions = len(set(partitions))
    is_cross = 1 if num_distinct_partitions > 1 else 0

    # 3. Aggregations & Order By
    has_agg = 1 if any(f in cypher_upper for f in ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]) else 0
    has_order_by = 1 if "ORDER BY" in cypher_upper else 0
    has_limit = 1 if "LIMIT" in cypher_upper else 0

    # 4. Length and variable-length traversal
    query_length = len(cypher)
    has_varlen = 1 if "*" in cypher else 0

    # Label proxy: simulated network hop cost
    # Target: log(1 + network_cost_units)
    simulated_cost = (num_distinct_partitions - 1) * 50.0 + (hops * 10.0) + (100.0 if has_agg else 10.0)
    target_label = np.log1p(simulated_cost)

    return {
        "hops": hops,
        "num_distinct_partitions": num_distinct_partitions,
        "is_cross_partition": is_cross,
        "has_agg": has_agg,
        "has_order_by": has_order_by,
        "has_limit": has_limit,
        "query_length": query_length,
        "has_varlen": has_varlen,
        "target_cost": target_label
    }


def main():
    print("=== Training Cross-Partition Cost Estimator (LightGBM) ===")
    data_dir = "/kaggle/input/text2gql-partition-data"
    if not os.path.exists(data_dir):
        data_dir = "data/output"

    records = []
    for domain in ["movie", "nba", "company"]:
        fp = os.path.join(data_dir, f"{domain}_augmented.jsonl")
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    records.append(json.loads(line))

    print(f"Loaded {len(records)} records.")

    features_list = [extract_ast_features(r) for r in records]
    df = pd.DataFrame(features_list)

    X = df.drop(columns=["target_cost"])
    y = df["target_cost"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = lgb.LGBMRegressor(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\nModel Evaluation Metrics:")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R^2  : {r2:.4f}")

    output_dir = "/kaggle/working/cost_estimator"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "lightgbm_cost_estimator.joblib")
    joblib.dump(model, model_path)
    print(f"\nModel saved successfully to {model_path} ✅")


if __name__ == "__main__":
    main()

