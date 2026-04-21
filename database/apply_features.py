import sqlite3

DB_PATH = "database/loaniq.db"
SQL_PATH = "sql/feature_engineering.sql"

conn = sqlite3.connect(DB_PATH)

with open(SQL_PATH) as f:
    sql = f.read()

conn.executescript(sql)
conn.commit()

# Spot check
row = conn.execute("""
    SELECT debt_to_income, annuity_to_income, age_years, ext_score_sum
    FROM model_features
    WHERE TARGET IS NOT NULL
    LIMIT 5
""").fetchall()

print("Sample rows from model_features:")
for r in row:
    print(r)

conn.close()
print("✅ Feature view created")