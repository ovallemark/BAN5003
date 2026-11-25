import os

import numpy as np
import pandas as pd
import psycopg2
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dotenv import load_dotenv

load_dotenv("module3regressioncredentials.env")

SEED = 42

DSN = {
    "host": "localhost",
    "port": 5432,
    "dbname": "BAN5003",
    "user": "postgres",
    "password": os.getenv("DB_PASSWORD"),
}

QUERY = """
WITH order_item_summary AS (
    SELECT
        oi.order_id,
        SUM(oi.freight_value) AS total_freight_value,
        AVG(p.product_weight_g) / 1000.0 AS avg_product_weight_kg,
        COUNT(*) AS item_count,
        COUNT(DISTINCT p.product_category_name) AS category_count
    FROM olist.order_items oi
    JOIN olist.products p ON oi.product_id = p.product_id
    GROUP BY oi.order_id
    HAVING COUNT(p.product_weight_g) = COUNT(*)
       AND BOOL_AND(oi.freight_value > 0)
),
order_review_summary AS (
    SELECT
        order_id,
        AVG(review_score) AS avg_review_score
    FROM olist.order_reviews
    GROUP BY order_id
)
SELECT
    CASE WHEN r.avg_review_score <= 2 THEN 1 ELSE 0 END AS low_review,
    i.total_freight_value,
    i.avg_product_weight_kg,
    i.item_count,
    i.category_count,
    c.customer_state
FROM olist.orders o
JOIN order_review_summary r ON o.order_id = r.order_id
JOIN order_item_summary i ON o.order_id = i.order_id
JOIN olist.customers c ON o.customer_id = c.customer_id
WHERE o.order_status = 'delivered';
"""


def main() -> None:
    with psycopg2.connect(**DSN) as conn:
        df = pd.read_sql(QUERY, conn)

    y = df["low_review"].values
    X = df.drop(columns=["low_review"])

    # 60/20/20 train/validation/test
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, random_state=SEED, stratify=y_tv
    )

    numeric_cols = [
        "total_freight_value",
        "avg_product_weight_kg",
        "item_count",
        "category_count",
    ]
    cat_cols = ["customer_state"]

    # Model A: Logistic Regression
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    pipe = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])

    pipe.fit(X_train, y_train)

    y_score = pipe.predict_proba(X_val)[:, 1]
    y_pred = (y_score >= 0.25).astype(int)

    cm = confusion_matrix(y_val, y_pred)
    print("Model A - Logistic Regression")
    print("Validation confusion matrix (rows=actual, cols=predicted):")
    print(f"               pred 0    pred 1")
    print(f"  actual 0  {cm[0,0]:>8d}  {cm[0,1]:>8d}")
    print(f"  actual 1  {cm[1,0]:>8d}  {cm[1,1]:>8d}")
    print()
    print(f"Validation precision : {precision_score(y_val, y_pred, zero_division=0):.3f}")
    print(f"Validation recall    : {recall_score(y_val, y_pred):.3f}")
    print(f"Validation AUC       : {roc_auc_score(y_val, y_score):.3f}")
    print()

    # Model B: Random Forest
    pre_rf = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    rf_pipe = Pipeline([
        ("pre", pre_rf),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=SEED,
        )),
    ])

    rf_pipe.fit(X_train, y_train)

    rf_score = rf_pipe.predict_proba(X_val)[:, 1]
    rf_pred = (rf_score >= 0.25).astype(int)

    rf_cm = confusion_matrix(y_val, rf_pred)
    print("Model B - Random Forest")
    print("Validation confusion matrix (rows=actual, cols=predicted):")
    print(f"               pred 0    pred 1")
    print(f"  actual 0  {rf_cm[0,0]:>8d}  {rf_cm[0,1]:>8d}")
    print(f"  actual 1  {rf_cm[1,0]:>8d}  {rf_cm[1,1]:>8d}")
    print()
    print(f"Validation precision : {precision_score(y_val, rf_pred, zero_division=0):.3f}")
    print(f"Validation recall    : {recall_score(y_val, rf_pred):.3f}")
    print(f"Validation AUC       : {roc_auc_score(y_val, rf_score):.3f}")
    print()
    #"Test set is locked until model selection is complete.\n"
    #"Re-run on the test set ONLY when comparing the final candidate."
    # Final Model - Random Forest on the Test Set
    rf_pipe.fit(X_tv, y_tv)

    rf_test_score = rf_pipe.predict_proba(X_test)[:, 1]
    rf_test_pred = (rf_test_score >= 0.25).astype(int)

    rf_test_cm = confusion_matrix(y_test, rf_test_pred)

    print()
    print("FINAL MODEL - RANDOM FOREST TEST RESULTS")
    print("Test confusion matrix (rows=actual, cols=predicted):")
    print(f"               pred 0    pred 1")
    print(f"  actual 0  {rf_test_cm[0,0]:>8d}  {rf_test_cm[0,1]:>8d}")
    print(f"  actual 1  {rf_test_cm[1,0]:>8d}  {rf_test_cm[1,1]:>8d}")
    print()
    print(f"Test precision : {precision_score(y_test, rf_test_pred, zero_division=0):.3f}")
    print(f"Test recall    : {recall_score(y_test, rf_test_pred):.3f}")
    print(f"Test AUC       : {roc_auc_score(y_test, rf_test_score):.3f}")


if __name__ == "__main__":
    main()