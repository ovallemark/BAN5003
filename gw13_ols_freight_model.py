"""BAN 5003 - Guided Walkthrough 13: OLS Regression: Freight Model

HOW TO USE THIS FILE
  1. Save it into your course project folder (the one with your .venv).
  2. Make sure these packages are installed in that venv (one time):
         pip install pandas psycopg2-binary statsmodels
  3. Edit the two lines marked  # <-- EDIT  below so they match the
     username and password you use in DBeaver (the default Postgres
     user is "postgres"; the password is whatever you set at install).
     The database name must be the one that contains the olist schema.
  4. Run it from a terminal in that folder with the venv active:
         python gw13_ols_freight_model.py
  Follow along with the Guided Walkthrough page in Canvas - it explains
  every block of this script.
"""
import os

import pandas as pd
import psycopg2
import statsmodels.formula.api as smf

from dotenv import load_dotenv

load_dotenv("module3regressioncredentials.env")

DSN = {
    "host": "localhost",
    "port": 5432,
    "dbname": "BAN5003",          # <-- EDIT if your database is named differently (e.g. "BAN5003")
    "user": "postgres",        # <-- EDIT: your Postgres username
    "password": os.getenv("DB_PASSWORD"),   # <-- EDIT: your Postgres password
}

QUERY = """
SELECT
    oi.freight_value,
    p.product_weight_g / 1000.0 AS product_weight_kg,
    c.customer_state
FROM olist.order_items oi
JOIN olist.products p ON oi.product_id = p.product_id
JOIN olist.orders o ON oi.order_id = o.order_id
JOIN olist.customers c ON o.customer_id = c.customer_id
WHERE p.product_weight_g IS NOT NULL
  AND oi.freight_value > 0;
"""


def main() -> None:
    with psycopg2.connect(**DSN) as conn:
        df = pd.read_sql(QUERY, conn)

    print(f"n rows fitted: {len(df):,}")

    # Fit: freight_value as a function of product_weight_kg and customer_state.
    # customer_state is categorical; SP becomes the reference level by default.
    model = smf.ols(
        "freight_value ~ product_weight_kg + C(customer_state, Treatment(reference='SP'))",
        data=df,
    ).fit()

    print()
    print(model.summary())

    print()
    print("Headline interpretations:")
    weight_coef = model.params["product_weight_kg"]
    weight_ci = model.conf_int().loc["product_weight_kg"]
    print(
        f"  Each additional kg of product weight is associated with "
        f"about {weight_coef:.2f} BRL more freight "
        f"(95% CI [{weight_ci[0]:.2f}, {weight_ci[1]:.2f}]), "
        "holding customer state constant."
    )
    print(f"  R-squared: {model.rsquared:.3f}")


if __name__ == "__main__":
    main()
