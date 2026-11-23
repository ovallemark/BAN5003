"""BAN 5003 - Guided Walkthrough 15: Your First Classifier (scikit-learn)

HOW TO USE THIS FILE
  1. Save it into your course project folder (the one with your .venv).
  2. Make sure these packages are installed in that venv (one time):
         pip install pandas numpy psycopg2-binary scikit-learn
  3. Edit the two lines marked  # <-- EDIT  below so they match the
     username and password you use in DBeaver (the default Postgres
     user is "postgres"; the password is whatever you set at install).
     The database name must be the one that contains the olist schema.
  4. Run it from a terminal in that folder with the venv active:
         python gw15_first_classifier.py
  Follow along with the Guided Walkthrough page in Canvas - it explains
  every block of this script.
"""
import os

import numpy as np
import pandas as pd
import psycopg2
from sklearn.compose import ColumnTransformer
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
    "dbname": "BAN5003",          # <-- EDIT if your database is named differently (e.g. "BAN5003")
    "user": "postgres",        # <-- EDIT: your Postgres username
    "password": os.getenv("DB_PASSWORD"),   # <-- EDIT: your Postgres password
}

QUERY = """
SELECT
    CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END AS low_review,
    oi.freight_value,
    p.product_weight_g / 1000.0 AS product_weight_kg,
    DATE_PART('day',
        o.order_delivered_customer_date - o.order_purchase_timestamp
    ) AS delivery_days,
    c.customer_state,
    p.product_category_name
FROM olist.orders o
JOIN olist.order_reviews r ON o.order_id = r.order_id
JOIN olist.order_items oi  ON o.order_id = oi.order_id
JOIN olist.products p      ON oi.product_id = p.product_id
JOIN olist.customers c     ON o.customer_id = c.customer_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND p.product_weight_g IS NOT NULL
  AND oi.freight_value > 0;
"""


def main() -> None:
    with psycopg2.connect(**DSN) as conn:
        df = pd.read_sql(QUERY, conn)

    df = df.dropna(subset=["delivery_days", "product_category_name"])
    df["delivery_days"] = df["delivery_days"].astype(float)
    y = df["low_review"].values
    X = df.drop(columns=["low_review"])

    # 60/20/20 train/validation/test
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, random_state=SEED, stratify=y_tv
    )

    numeric_cols = ["freight_value", "product_weight_kg", "delivery_days"]
    cat_cols = ["customer_state", "product_category_name"]

    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    pipe = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_val)
    y_score = pipe.predict_proba(X_val)[:, 1]

    cm = confusion_matrix(y_val, y_pred)
    print("Validation confusion matrix (rows=actual, cols=predicted):")
    print(f"               pred 0    pred 1")
    print(f"  actual 0  {cm[0,0]:>8d}  {cm[0,1]:>8d}")
    print(f"  actual 1  {cm[1,0]:>8d}  {cm[1,1]:>8d}")
    print()
    print(f"Validation precision : {precision_score(y_val, y_pred):.3f}")
    print(f"Validation recall    : {recall_score(y_val, y_pred):.3f}")
    print(f"Validation AUC       : {roc_auc_score(y_val, y_score):.3f}")
    print()
    print(
        "Test set is locked until model selection is complete.\n"
        "Re-run on the test set ONLY when comparing the final candidate.")


if __name__ == "__main__":
    main()
