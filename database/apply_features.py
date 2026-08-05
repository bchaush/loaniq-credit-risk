import sqlite3

DB_PATH = "database/loaniq.db"
SQL_PATH = "sql/feature_engineering.sql"

conn = sqlite3.connect(DB_PATH)
conn.execute("DROP VIEW IF EXISTS model_features")

with open(SQL_PATH, encoding="utf-8") as f:
    sql = f.read()

conn.executescript(sql)
conn.commit()

row = conn.execute("""
    SELECT debt_to_income, annuity_to_income, age_years, employed_years,
           employment_to_age_ratio, is_unemployed, ext_score_sum
    FROM model_features
    WHERE TARGET IS NOT NULL
    LIMIT 5
""").fetchall()

print("Sample rows from model_features:")
for r in row:
    print(r)

conn.close()
print("OK Feature view created")
