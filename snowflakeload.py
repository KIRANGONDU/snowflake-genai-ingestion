import snowflake.connector
import pandas as pd
 
 
SNOWFLAKE_USER = "Saikiran"
SNOWFLAKE_PASSWORD = "Gondusaikiran@2003"
SNOWFLAKE_ACCOUNT = "TRUSCGX-ZB98835"
SNOWFLAKE_WAREHOUSE = "COMPUTE_WH"
SNOWFLAKE_DATABASE = "dbt"
SNOWFLAKE_SCHEMA = "kiran"
 
 
csv_path = "staging_data/transformed_file.csv"
df = pd.read_csv(csv_path, dtype=str)
 
df["dob"] = pd.to_datetime(df["dob"], format="%Y%m%d", errors="coerce") \
              .dt.strftime("%Y-%m-%d")
df = df.where(pd.notnull(df), None)
 
conn = snowflake.connector.connect(
    user=SNOWFLAKE_USER,
    password=SNOWFLAKE_PASSWORD,
    account=SNOWFLAKE_ACCOUNT,
    warehouse=SNOWFLAKE_WAREHOUSE,
    database=SNOWFLAKE_DATABASE,
    schema=SNOWFLAKE_SCHEMA
)
 
cursor = conn.cursor()
 
insert_sql = """
INSERT INTO CUSTOMER_MASTER1(
    customer_id,
    firstname,
    lastname,
    gender,
    dob,
    email,
    phone,
    city,
    country,
    load_date
)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
"""
 
for _, row in df.iterrows():
    cursor.execute(insert_sql, tuple(row))
 
conn.commit()
 
cursor.close()
conn.close()
 
print("Data loaded into Snowflake using Option (Python Direct Load)")
 