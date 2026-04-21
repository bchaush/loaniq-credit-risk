import pandas as pd
import sqlite3
import os

# Paths
DATA_PATH = "data/application_train.csv"
DB_PATH = "database/loaniq.db"

os.makedirs("database", exist_ok=True)

print("Loading CSV...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

# Keep only the columns we need for feature engineering + target
COLS = [
    "SK_ID_CURR",
    "TARGET",                    # 1 = defaulted, 0 = repaid
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "DAYS_BIRTH",                # negative = days before application
    "DAYS_EMPLOYED",             # negative = days before application
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "EXT_SOURCE_1",              # external credit scores (very predictive)
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "REGION_RATING_CLIENT",
    "REG_CITY_NOT_WORK_CITY",
    "FLAG_DOCUMENT_3",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
]

df = df[COLS].copy()
print(f"Trimmed to {len(df.columns)} columns")

# Basic cleaning
df["DAYS_BIRTH"] = df["DAYS_BIRTH"].abs()         # convert to positive age in days
df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].apply(
    lambda x: x if x < 0 else None               # 365243 = unemployed/retired sentinel
).abs()

df["FLAG_OWN_CAR"] = (df["FLAG_OWN_CAR"] == "Y").astype(int)
df["FLAG_OWN_REALTY"] = (df["FLAG_OWN_REALTY"] == "Y").astype(int)

print("Writing to SQLite...")
conn = sqlite3.connect(DB_PATH)
df.to_sql("applications", conn, if_exists="replace", index=False)

# Verify
count = conn.execute("SELECT COUNT(*) FROM applications").fetchone()[0]
default_rate = conn.execute("SELECT AVG(TARGET) FROM applications").fetchone()[0]
print(f"✅ Database built: {count:,} rows")
print(f"✅ Default rate: {default_rate:.2%}")
conn.close()