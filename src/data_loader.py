"""
data_loader.py
--------------
Loads and preprocesses M5 Walmart sales data for the CA_1 store.

Expects the following files in data/raw/:
    - sales_train_validation.csv
    - calendar.csv
    - sell_prices.csv

Returns a clean daily-aggregated DataFrame for CA_1.
"""

import os
import pandas as pd

# Resolve paths relative to project root regardless of where script is called from
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "raw")


def _get_path(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three M5 source files and return them as raw DataFrames."""
    print("Loading sales_train_validation.csv ...")
    sales = pd.read_csv(_get_path("sales_train_validation.csv"))

    print("Loading calendar.csv ...")
    calendar = pd.read_csv(_get_path("calendar.csv"))

    print("Loading sell_prices.csv ...")
    prices = pd.read_csv(_get_path("sell_prices.csv"))

    return sales, calendar, prices


def filter_ca1(sales: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows belonging to store CA_1."""
    mask = sales["store_id"] == "CA_1"
    ca1 = sales[mask].copy()
    print(f"CA_1 rows: {len(ca1):,}  (of {len(sales):,} total)")
    return ca1


def melt_to_long(ca1: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the wide sales matrix (one column per day) into long format.

    Result columns: id, item_id, dept_id, cat_id, store_id, state_id, d, sales
    """
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in ca1.columns if c.startswith("d_")]

    long = ca1.melt(id_vars=id_cols, value_vars=day_cols, var_name="d", value_name="sales")
    return long


def aggregate_daily(long: pd.DataFrame) -> pd.DataFrame:
    """Sum all item-level sales to a single daily total for CA_1."""
    daily = long.groupby("d", as_index=False)["sales"].sum()
    return daily


def merge_calendar(daily: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """
    Merge daily sales with calendar metadata.

    Adds: date, event_name_1, snap_CA.
    """
    cal_cols = ["d", "date", "event_name_1", "snap_CA"]
    cal_subset = calendar[cal_cols].copy()
    merged = daily.merge(cal_subset, on="d", how="left")
    return merged


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive time-based and categorical features from the date column.

    Adds:
        day_of_week  : 0 = Monday … 6 = Sunday
        month        : 1–12
        year         : integer year
        is_weekend   : bool  (Saturday or Sunday)
        is_holiday   : bool  (event_name_1 is not null/empty)
        snap_day     : bool  (snap_CA == 1)
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = df["day_of_week"].isin([5, 6])
    df["is_holiday"] = df["event_name_1"].notna() & (df["event_name_1"].str.strip() != "")
    df["snap_day"] = df["snap_CA"].fillna(0).astype(bool)

    return df


def select_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Return final clean DataFrame with canonical column order, sorted by date."""
    final_cols = [
        "date", "sales", "day_of_week", "month", "year",
        "is_weekend", "is_holiday", "snap_day",
    ]
    return df[final_cols].sort_values("date").reset_index(drop=True)


def load_ca1_daily() -> pd.DataFrame:
    """
    Master function — run the full pipeline and return the clean CA_1 daily DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: date, sales, day_of_week, month, year, is_weekend, is_holiday, snap_day
    """
    sales, calendar, prices = load_raw_data()
    ca1 = filter_ca1(sales)
    long = melt_to_long(ca1)
    daily = aggregate_daily(long)
    merged = merge_calendar(daily, calendar)
    featured = engineer_features(merged)
    clean = select_and_sort(featured)
    return clean


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_ca1_daily()
    print("\n--- Shape ---")
    print(df.shape)
    print("\n--- Head ---")
    print(df.head(10))
    print("\n--- dtypes ---")
    print(df.dtypes)
    print("\n--- Null counts ---")
    print(df.isnull().sum())
