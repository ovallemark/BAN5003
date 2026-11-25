"""BAN 5003 - Comparing models you can defend (Week 13 model-comparison lecture).

THE BUSINESS QUESTION (deliberately NOT one of the M4 report questions)
  When an order is placed, Olist's seller-operations team can send the seller
  a reminder (a "nudge"). They want to nudge the orders most likely to be
  handed to the carrier AFTER the seller's shipping deadline: a "late
  dispatch". About 9% of orders are dispatched late. Which model should rank
  the orders, and how many should get a nudge?

WHAT THIS SCRIPT DOES - same data, same split, same metrics for every model
  PART 1  Scoreboard: four models scored on one validation set
  PART 2  Is the gap real? 5-fold cross-validation: the mean AND the spread
  PART 3  Which model saves the most money? Nudge cost vs missed late orders
  PART 4  The regression version: the metric you pick decides the winner
  FINAL   The chosen model on the test set - used exactly once

HOW TO USE THIS FILE
  1. Save it in your project folder (the one with your .venv).
  2. One time, with the venv active:
         pip install pandas psycopg2-binary scikit-learn
  3. Edit the three lines marked  # <-- EDIT  (same values as DBeaver).
  4. Run every part (about 1 minute):   python m4cmp_compare_models.py
     or one part, for example part 2:   python m4cmp_compare_models.py 2

PYTHON YOU WILL SEE IN THIS FILE (quick reference)
  df                         a pandas DataFrame: one row per order
  model.fit(X, y)            trains a model on features X and answers y
  model.predict_proba(X)     a risk score (probability) for each order
  # text after a hash        a comment: Python ignores it; it is there for you
  f"{x:.3f}"                 x printed with 3 decimals
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
import psycopg2
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")      # hides harmless pandas/sklearn notices

DSN = {
    "host": "localhost",
    "port": 5432,
    "dbname": "olist",          # <-- EDIT if your database is named differently (e.g. "BAN5003")
    "user": "postgres",        # <-- EDIT: your Postgres username
    "password": "CHANGE_ME",   # <-- EDIT: your Postgres password
}
SEED = 42                      # fixes the random split so everyone gets the same numbers
NUDGE_SHARE = 0.10             # part 1: the team can nudge 10% of orders
NUDGE_COST = 2.0               # part 3: staff time for one nudge, in reais (R$)
MISS_COSTS = (10.0, 20.0, 40.0)  # part 3: what one missed late dispatch costs, R$

# One row per order. Every feature is known WHEN THE ORDER IS PLACED.
# days_to_carrier is only known after dispatch: it is part 4's OUTCOME and is
# never used as a feature (it would leak the answer - see the Methods Primer).
SQL = """
WITH items AS (
    SELECT oi.order_id,
           MAX(oi.shipping_limit_date)                    AS limit_date,
           SUM(oi.price)                                  AS price,
           SUM(oi.freight_value)                          AS freight,
           COUNT(*)                                       AS n_items,
           SUM(COALESCE(p.product_weight_g, 0)) / 1000.0  AS weight_kg,
           COUNT(DISTINCT oi.seller_id)                   AS n_sellers,
           MAX(CASE WHEN s.seller_state = 'SP' THEN 1 ELSE 0 END) AS sp_seller
    FROM olist.order_items oi
    JOIN olist.products p ON oi.product_id = p.product_id
    JOIN olist.sellers  s ON oi.seller_id  = s.seller_id
    GROUP BY oi.order_id
)
SELECT CASE WHEN o.order_delivered_carrier_date > i.limit_date THEN 1 ELSE 0 END AS dispatch_late,
       i.price, i.freight, i.n_items, i.weight_kg, i.n_sellers, i.sp_seller,
       DATE_PART('day', i.limit_date - o.order_purchase_timestamp) AS handling_days,
       EXTRACT(DOW  FROM o.order_purchase_timestamp)               AS dow,
       EXTRACT(HOUR FROM o.order_purchase_timestamp)               AS hr,
       CASE WHEN EXTRACT(MONTH FROM o.order_purchase_timestamp) IN (11, 12)
            THEN 1 ELSE 0 END                                      AS holiday,
       DATE_PART('day', o.order_estimated_delivery_date - o.order_purchase_timestamp) AS promised_days,
       EXTRACT(EPOCH FROM o.order_delivered_carrier_date - o.order_purchase_timestamp)
           / 86400.0                                               AS days_to_carrier
FROM olist.orders o
JOIN items i ON o.order_id = i.order_id
WHERE o.order_delivered_carrier_date IS NOT NULL;
"""
FEATURES = ["price", "freight", "n_items", "weight_kg", "n_sellers", "sp_seller",
            "handling_days", "dow", "hr", "holiday", "promised_days"]


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def heading(part: str, title: str) -> None:
    line = "=" * 74
    print(f"\n{line}\n{part}  |  {title}\n{line}")


def load_data() -> pd.DataFrame:
    conn = psycopg2.connect(**DSN)
    try:
        return pd.read_sql(SQL, conn).dropna()
    finally:
        conn.close()


def candidate_models() -> dict:
    """The four candidates. Same features, same training rows for every one."""
    return {
        # Weighted sum of the features. Scaling first helps logistic regression.
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        # Up to 8 levels of yes/no questions (depth chosen in W11 to avoid overfitting).
        "Decision tree (depth 8)": DecisionTreeClassifier(max_depth=8, random_state=SEED),
        # 300 trees on random samples, averaged.
        "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                                n_jobs=-1, random_state=SEED),
        # Many small trees, each correcting the last one's mistakes.
        "Gradient boosting": HistGradientBoostingClassifier(random_state=SEED),
    }


def split(df: pd.DataFrame):
    """60% train / 20% validation / 20% test. stratify keeps 9% late in each."""
    X, y = df[FEATURES], df["dispatch_late"]
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=SEED)
    return X_train, X_val, X_test, y_train, y_val, y_test


def top_share_stats(y_true: np.ndarray, score: np.ndarray, share: float) -> tuple:
    """Nudge the riskiest `share` of orders. Return precision, recall and lift.

    precision = of the nudged orders, the share that really were late
    recall    = of all late orders, the share we nudged
    lift      = precision / overall late rate (how many times better than random)
    """
    n_flag = int(round(len(score) * share))
    flagged = np.argsort(-score)[:n_flag]          # positions of the riskiest orders
    caught = y_true[flagged].sum()
    precision = caught / n_flag
    recall = caught / y_true.sum()
    return precision, recall, precision / y_true.mean()


# ---------------------------------------------------------------------------
# PART 1: the scoreboard
# ---------------------------------------------------------------------------
def part1(df, data, fitted) -> None:
    heading("PART 1", "Scoreboard: four models, one validation set, same metrics")
    X_train, X_val, X_test, y_train, y_val, y_test = data
    print(f"Orders: {len(df):,} | late dispatch rate: {df['dispatch_late'].mean():.1%} | "
          f"train {len(X_train):,} / validation {len(X_val):,} / test {len(X_test):,}")
    print(f"\n{'model':<25}{'fit (s)':>8}{'val AUC':>9}{'precision':>11}{'recall':>8}"
          f"{'lift':>6}   (nudging the riskiest {NUDGE_SHARE:.0%})")
    val_scores = {}
    for name, (model, seconds) in fitted.items():
        score = model.predict_proba(X_val)[:, 1]      # column 1 = chance of "late"
        val_scores[name] = score
        p, r, lift = top_share_stats(y_val.values, score, NUDGE_SHARE)
        print(f"{name:<25}{seconds:>8.1f}{roc_auc_score(y_val, score):>9.3f}"
              f"{p:>11.1%}{r:>8.1%}{lift:>6.2f}")
    print(f"\nAccuracy trap: a 'model' that says ON TIME for every order is "
          f"{1 - y_val.mean():.1%} accurate and catches 0 late orders.")
    print("""
HOW TO READ THIS
  AUC ranks ALL orders; precision/recall/lift describe only the orders the
  team would actually nudge. A random pick of 10% would catch 10% of the
  late orders (lift 1.0). Lift 2.3 = the model's nudge list has 2.3 times
  as many late orders as a random list. Same data, same split, same metrics
  for every model: that is what makes the comparison fair.""")
    return val_scores


# ---------------------------------------------------------------------------
# PART 2: is the gap real?
# ---------------------------------------------------------------------------
def part2(data) -> dict:
    heading("PART 2", "Is the gap real? 5-fold cross-validation on the training rows")
    X_train, _, _, y_train, _, _ = data
    # The SAME five folds for every model (random_state fixes them), so we can
    # compare the models fold by fold: a paired comparison.
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv = {}
    print(f"{'model':<25}{'fold 1':>8}{'fold 2':>8}{'fold 3':>8}{'fold 4':>8}{'fold 5':>8}"
          f"{'mean':>8}{'SD':>7}")
    for name, model in candidate_models().items():
        scores = cross_val_score(model, X_train, y_train, cv=folds, scoring="roc_auc")
        cv[name] = scores
        print(f"{name:<25}" + "".join(f"{s:>8.3f}" for s in scores)
              + f"{scores.mean():>8.3f}{scores.std():>7.3f}")

    print("\nHead to head, fold by fold (same folds, so the differences are fair):")
    pairs = [("Decision tree (depth 8)", "Logistic regression"),
             ("Gradient boosting", "Logistic regression"),
             ("Gradient boosting", "Random forest")]
    for a, b in pairs:
        diff = cv[a] - cv[b]                    # one difference per fold
        wins = int((diff > 0).sum())
        print(f"  {a} minus {b}: average gap {diff.mean():+.3f} "
              f"(spread {diff.std():.3f}); {a} wins {wins} of 5 folds")
    print("""
HOW TO READ THIS
  Each fold is a different 20% of the training rows held out for scoring.
  SD (standard deviation) is how much a model's score moves from fold to
  fold: that is the noise. A gap SMALLER than the noise is a tie. A gap
  several times the noise, won in all 5 folds, is real. Ties are settled on
  something else: simplicity, explainability, speed, or cost (part 3).""")
    return cv


# ---------------------------------------------------------------------------
# PART 3: which model saves the most money?
# ---------------------------------------------------------------------------
def cost_curve(y_true: np.ndarray, score: np.ndarray, miss_cost: float) -> tuple:
    """Total cost when we nudge the riskiest 0%, 0.5%, 1%, ... 100% of orders."""
    order = np.argsort(-score)                           # riskiest first
    late_sorted = y_true[order]
    caught = np.concatenate([[0], np.cumsum(late_sorted)])  # late orders caught after k nudges
    n_nudged = np.arange(len(y_true) + 1)
    missed = y_true.sum() - caught
    total = NUDGE_COST * n_nudged + miss_cost * missed
    shares = np.linspace(0, 1, 201)                      # 0%, 0.5%, ..., 100%
    idx = np.round(shares * len(y_true)).astype(int)
    return shares, total[idx]


def part3(data, val_scores) -> dict:
    heading("PART 3", "Which model saves the most money?")
    y_val = data[4].values
    n, n_late = len(y_val), int(y_val.sum())
    print(f"Validation orders: {n:,}; late: {n_late:,}. One nudge costs R${NUDGE_COST:.0f}.")
    out = {}
    for miss_cost in MISS_COSTS:
        no_model = miss_cost * n_late                    # nudge nobody: pay for every miss
        nudge_all = NUDGE_COST * n                       # nudge everyone: no misses
        print(f"\nA missed late dispatch costs R${miss_cost:.0f} "
              f"({miss_cost / NUDGE_COST:.0f} times a nudge)")
        print(f"  nudge nobody: R${no_model:,.0f}   nudge everyone: R${nudge_all:,.0f}")
        for name, score in val_scores.items():
            shares, total = cost_curve(y_val, score, miss_cost)
            best = int(np.argmin(total))
            out[(name, miss_cost)] = (shares[best], total[best])
            print(f"  {name:<25} best: nudge the riskiest {shares[best]:>6.1%} "
                  f"-> R${total[best]:>8,.0f}  (saves {1 - total[best] / min(no_model, nudge_all):.1%} "
                  f"vs the better no-model policy)")
    print("""
HOW TO READ THIS
  The costs are ASSUMPTIONS: replace them with your sponsor's numbers. The
  model and the threshold are chosen together. When a miss is cheap, the
  best plan nudges few orders; when a miss is expensive, the best plan
  nudges nearly everyone and the model hardly matters. Models earn their
  keep in between.""")
    return out


# ---------------------------------------------------------------------------
# PART 4: the regression version - the metric picks the winner
# ---------------------------------------------------------------------------
def part4(df, data) -> dict:
    heading("PART 4", "Regression: predict DAYS to dispatch - the metric picks the winner")
    X_train, X_val = data[0], data[1]
    days = df["days_to_carrier"]
    # A few orders show the carrier pickup BEFORE the purchase (data errors): drop them.
    tr = X_train.index[days.loc[X_train.index] >= 0]
    va = X_val.index[days.loc[X_val.index] >= 0]
    Xtr, ytr = df.loc[tr, FEATURES], days.loc[tr]
    Xva, yva = df.loc[va, FEATURES], days.loc[va]
    print(f"Days from purchase to carrier pickup: mean {ytr.mean():.2f}, "
          f"median {ytr.median():.2f} (a long right tail pulls the mean up)")

    models = {
        "Guess the median (no model)": None,
        "Linear regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Boosting, squared error": HistGradientBoostingRegressor(random_state=SEED),
        "Boosting, absolute error": HistGradientBoostingRegressor(loss="absolute_error",
                                                                  random_state=SEED),
    }
    out = {}
    print(f"\n{'model':<30}{'MAE (days)':>11}{'RMSE (days)':>12}{'R-squared':>11}")
    for name, model in models.items():
        if model is None:
            pred = np.full(len(yva), ytr.median())
        else:
            pred = model.fit(Xtr, ytr).predict(Xva)
        mae = mean_absolute_error(yva, pred)
        rmse = mean_squared_error(yva, pred) ** 0.5
        out[name] = (mae, rmse, r2_score(yva, pred))
        print(f"{name:<30}{mae:>11.2f}{rmse:>12.2f}{out[name][2]:>11.3f}")
    print("""
HOW TO READ THIS
  MAE = the typical miss, in days. RMSE squares each miss first, so a few
  big misses count a lot. Squared-error boosting wins RMSE; absolute-error
  boosting wins MAE. Neither is "the best model" until the business says
  which mistake hurts: a steady small miss (MAE) or a rare huge one (RMSE).
  Pick the metric FIRST, then the model.""")
    return out


# ---------------------------------------------------------------------------
# FINAL: the chosen model on the test set, once
# ---------------------------------------------------------------------------
def final_test(data, fitted, chosen: str, share: float) -> None:
    heading("FINAL", f"{chosen} on the test set (used exactly once)")
    X_test, y_test = data[2], data[5].values
    model = fitted[chosen][0]
    score = model.predict_proba(X_test)[:, 1]
    cut = np.quantile(score, 1 - share)            # nudge the riskiest `share` of orders
    flag = score >= cut
    tp = int((flag & (y_test == 1)).sum()); fp = int((flag & (y_test == 0)).sum())
    fn = int((~flag & (y_test == 1)).sum()); tn = int((~flag & (y_test == 0)).sum())
    print(f"Test AUC: {roc_auc_score(y_test, score):.3f}   "
          f"(threshold chosen on VALIDATION: nudge the riskiest {share:.1%})")
    print(f"\n                 predicted late   predicted on time")
    print(f"  actually late  {tp:>14,}   {fn:>17,}")
    print(f"  on time        {fp:>14,}   {tn:>17,}")
    print(f"\n  precision {tp / (tp + fp):.1%} | recall {tp / (tp + fn):.1%}")
    print("""
HOW TO READ THIS
  The test rows were never used to pick the model or the threshold, so this
  is the honest number to put in the report. If it is far below the
  validation number, something leaked or overfit: do not go back and tune.""")


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------
def main() -> None:
    wanted = set(sys.argv[1:]) or {"1", "2", "3", "4", "final"}
    print("Loading orders from Postgres...")
    df = load_data()
    data = split(df)
    fitted = {}
    for name, model in candidate_models().items():
        start = time.time()
        model.fit(data[0], data[3])                 # train rows only
        fitted[name] = (model, time.time() - start)
    if "1" in wanted:
        val_scores = part1(df, data, fitted)
    else:
        val_scores = {n: m.predict_proba(data[1])[:, 1] for n, (m, _) in fitted.items()}
    if "2" in wanted:
        part2(data)
    costs = part3(data, val_scores) if "3" in wanted else None
    if "4" in wanted:
        part4(df, data)
    if "final" in wanted:
        chosen, middle = "Gradient boosting", MISS_COSTS[1]
        share = costs[(chosen, middle)][0] if costs else 0.20
        final_test(data, fitted, chosen, share)


if __name__ == "__main__":
    main()

