"""BAN 5003 - Guided Walkthrough 11: Confidence Interval for Average Order Value

HOW TO USE THIS FILE
  1. Save it into your course project folder (the one with your .venv).
  2. Make sure these packages are installed in that venv (one time):
         pip install pandas psycopg2-binary
  3. Edit the two lines marked  # <-- EDIT  below so they match the
     username and password you use in DBeaver (the default Postgres
     user is "postgres"; the password is whatever you set at install).
     The database name must be the one that contains the olist schema.
  4. Run it from a terminal in that folder with the venv active:
         python gw11_confidence_interval.py
  Follow along with the Guided Walkthrough page in Canvas - it explains
  every block of this script.
"""
import math
import os

import pandas as pd
import psycopg2
from psycopg2 import sql


DSN = {
    "host": "localhost",
    "port": 5432,
    "dbname": "BAN5003",          # <-- EDIT if your database is named differently (e.g. "BAN5003")
    "user": "postgres",        # <-- EDIT: your Postgres username
    "password": "***REMOVED***",   # <-- EDIT: your Postgres password
}

QUERY = """
WITH order_totals AS (
    SELECT
        order_id,
        SUM(price + freight_value) AS order_total
    FROM olist.order_items
    GROUP BY order_id
)
SELECT order_id, order_total
FROM order_totals;
"""


def main() -> None:
    with psycopg2.connect(**DSN) as conn:
        df = pd.read_sql(QUERY, conn)

    n = len(df)
    mean = df["order_total"].mean()
    std = df["order_total"].std(ddof=1)
    se = std / math.sqrt(n)
    z = 1.96  # 95% normal critical value; n is large so this is fine
    margin = z * se
    ci_low = mean - margin
    ci_high = mean + margin

    print(f"n             : {n:,}")
    print(f"mean (BRL)    : {mean:.2f}")
    print(f"std (BRL)     : {std:.2f}")
    print(f"std error     : {se:.2f}")
    print(f"95% CI (BRL)  : [{ci_low:.2f}, {ci_high:.2f}]")
    print()
    print("Sponsor-ready sentence:")
    print(
        f"  The average order value in olist is about {mean:.0f} reais, "
        f"95% CI [{ci_low:.0f}, {ci_high:.0f}]."
    )


if __name__ == "__main__":
    main()
