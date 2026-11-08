"""BAN 5003 - Guided Walkthrough 12: t-Test: Late vs On-Time Review Scores

HOW TO USE THIS FILE
  1. Save it into your course project folder (the one with your .venv).
  2. Make sure these packages are installed in that venv (one time):
         pip install pandas psycopg2-binary scipy
  3. Edit the two lines marked  # <-- EDIT  below so they match the
     username and password you use in DBeaver (the default Postgres
     user is "postgres"; the password is whatever you set at install).
     The database name must be the one that contains the olist schema.
  4. Run it from a terminal in that folder with the venv active:
         python gw12_t_test_late_vs_ontime.py
  Follow along with the Guided Walkthrough page in Canvas - it explains
  every block of this script.
"""
import os

import pandas as pd
import psycopg2
from scipy import stats


DSN = {
    "host": "localhost",
    "port": 5432,
    "dbname": "BAN5003",          # <-- EDIT if your database is named differently (e.g. "BAN5003")
    "user": "postgres",        # <-- EDIT: your Postgres username
    "password": "***REMOVED***",   # <-- EDIT: your Postgres password
}

QUERY = """
SELECT
    CASE
        WHEN o.order_delivered_customer_date >
             o.order_estimated_delivery_date THEN 'late'
        ELSE 'on_time'
    END AS delivery_bucket,
    r.review_score
FROM olist.orders o
JOIN olist.order_reviews r ON o.order_id = r.order_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL;
"""


def main() -> None:
    with psycopg2.connect(**DSN) as conn:
        df = pd.read_sql(QUERY, conn)

    late = df.loc[df["delivery_bucket"] == "late", "review_score"]
    ontime = df.loc[df["delivery_bucket"] == "on_time", "review_score"]

    # Welch's t-test (scipy default — robust to unequal variance)
    t_stat, p_value = stats.ttest_ind(late, ontime, equal_var=False)
    mean_diff = late.mean() - ontime.mean()
    pooled_std = (((late.std() ** 2) + (ontime.std() ** 2)) / 2) ** 0.5
    cohens_d = mean_diff / pooled_std

    print(f"n (late)         : {len(late):,}")
    print(f"n (on time)      : {len(ontime):,}")
    print(f"mean review LATE : {late.mean():.2f}")
    print(f"mean review ON   : {ontime.mean():.2f}")
    print(f"mean diff (stars): {mean_diff:.2f}")
    print(f"Cohen's d        : {cohens_d:.2f}")
    print(f"t-statistic      : {t_stat:.2f}")
    print(f"p-value          : {p_value:.3g}")
    print()
    print("Sponsor-ready sentence:")
    print(
        f"  Late deliveries score about {abs(mean_diff):.1f} stars lower "
        f"than on-time deliveries (Cohen's d {cohens_d:.2f}; p < 0.001)."
    )


if __name__ == "__main__":
    main()
