"""BAN 5003 - Guided Walkthrough 14: Logistic Regression: Low-Review Orders

HOW TO USE THIS FILE
  1. Save it into your course project folder (the one with your .venv).
  2. Make sure these packages are installed in that venv (one time):
         pip install pandas numpy psycopg2-binary statsmodels
  3. Edit the two lines marked  # <-- EDIT  below so they match the
     username and password you use in DBeaver (the default Postgres
     user is "postgres"; the password is whatever you set at install).
     The database name must be the one that contains the olist schema.
  4. Run it from a terminal in that folder with the venv active:
         python gw14_logistic_low_review.py
  Follow along with the Guided Walkthrough page in Canvas - it explains
  every block of this script.
"""
import os

import numpy as np
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
    CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END AS low_review,
    oi.freight_value,
    p.product_weight_g / 1000.0 AS product_weight_kg,
    DATE_PART('day',
        o.order_delivered_customer_date - o.order_purchase_timestamp
    ) AS delivery_days
FROM olist.orders o
JOIN olist.order_reviews r ON o.order_id = r.order_id
JOIN olist.order_items oi  ON o.order_id = oi.order_id
JOIN olist.products p      ON oi.product_id = p.product_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND p.product_weight_g IS NOT NULL
  AND oi.freight_value > 0;
"""


def main() -> None:
    with psycopg2.connect(**DSN) as conn:
        df = pd.read_sql(QUERY, conn)

    df = df.dropna(subset=["delivery_days"])
    df["delivery_days"] = df["delivery_days"].astype(float)

    print(f"n rows fitted    : {len(df):,}")
    print(f"low_review rate  : {df['low_review'].mean():.3f}")

    model = smf.logit(
        "low_review ~ freight_value + product_weight_kg + delivery_days",
        data=df,
    ).fit(disp=False)

    print()
    print(model.summary())

    print()
    print("Odds ratios with 95% confidence intervals:")
    coef = model.params
    ci = model.conf_int()
    or_df = pd.DataFrame({
        "OR": np.exp(coef),
        "OR_lo": np.exp(ci[0]),
        "OR_hi": np.exp(ci[1]),
    })
    print(or_df.round(3))

    days_or = float(np.exp(coef["delivery_days"]))
    days_lo = float(np.exp(ci.loc["delivery_days", 0]))
    days_hi = float(np.exp(ci.loc["delivery_days", 1]))
    pct = (days_or - 1) * 100

    print()
    print("Sponsor-ready sentence:")
    print(
        f"  Each extra day of delivery is associated with about a "
        f"{pct:.1f}% increase in the odds of a low review "
        f"(OR {days_or:.3f}; 95% CI [{days_lo:.3f}, {days_hi:.3f}])."
    )


if __name__ == "__main__":
    main()
