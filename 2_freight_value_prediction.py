import pandas as pd
import psycopg2
import statsmodels.formula.api as smf

query = """
SELECT
    oi.freight_value,
    p.product_weight_g / 1000.0 AS product_weight_kg,
    p.product_length_cm,
    p.product_height_cm,
    p.product_width_cm,
    c.customer_state
FROM olist.order_items AS oi
JOIN olist.products AS p
    ON oi.product_id = p.product_id
JOIN olist.orders AS o
    ON oi.order_id = o.order_id
JOIN olist.customers AS c
    ON o.customer_id = c.customer_id
WHERE
    oi.freight_value >= 0
    AND p.product_weight_g > 0
    AND p.product_length_cm > 0
    AND p.product_height_cm > 0
    AND p.product_width_cm > 0
    AND c.customer_state IS NOT NULL;
"""

conn = psycopg2.connect(
    host="localhost",
    database="BAN5003",
    user="postgres",
    password="***REMOVED***",
    port="5432"
)

df = pd.read_sql(query, conn)
conn.close()

model = smf.ols(
    "freight_value ~ product_weight_kg + product_length_cm + product_height_cm + product_width_cm + C(customer_state, Treatment(reference='SP'))",
    data=df
).fit()

print(model.summary())

results = pd.DataFrame({
    "Estimate": model.params,
    "Std. Error": model.bse,
    "95% CI Lower": model.conf_int()[0],
    "95% CI Upper": model.conf_int()[1]
})

results.to_excel("freight_regression_results.xlsx")