
# load_to_snowflake.py
import os
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

DEFAULT_COLS = [
    "customer_id","firstname","lastname","gender","dob","email","phone","city","country","load_date"
]

def connect_snowflake(cfg: dict):
    """
    cfg keys: user, password, account, warehouse, database, schema
    """
    return snowflake.connector.connect(
        user=cfg.get("user"),
        password=cfg.get("password"),
        account=cfg.get("account"),
        warehouse=cfg.get("warehouse"),
        database=cfg.get("database"),
        schema=cfg.get("schema"),
    )

def ensure_table(conn, table_name: str, create_sql: str | None = None):
    """
    Create target table if not exists. If create_sql is None, use default DDL with canonical columns.
    """
    if create_sql is None:
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
          CUSTOMER_ID STRING,
          FIRSTNAME   STRING,
          LASTNAME    STRING,
          GENDER      STRING,
          DOB         DATE,
          EMAIL       STRING,
          PHONE       STRING,
          CITY        STRING,
          COUNTRY     STRING,
          LOAD_DATE   DATE
        );
        """
    with conn.cursor() as cur:
        cur.execute(create_sql)

def load_csv_to_df(csv_path: str, cols: list[str] = None) -> pd.DataFrame:
    """
    Reads CSV, normalizes blanks to None, ensures column order.
    """
    if cols is None:
        cols = DEFAULT_COLS
    df = pd.read_csv(csv_path, dtype=str)
    # Normalize blanks to None
    df = df.where(pd.notnull(df), None)

    # If your CSV has DOB like YYYYMMDD, uncomment to convert:
    # df["dob"] = pd.to_datetime(df["dob"], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")

    # Ensure column order matches table
    df = df[cols]
    return df

def run_load(csv_path: str, cfg: dict, table_name: str,
             create_if_not_exists: bool = True,
             create_sql: str | None = None) -> int:
    """
    Connects, optionally creates table, and bulk loads CSV into Snowflake using write_pandas.
    """
    conn = connect_snowflake(cfg)
    try:
        if create_if_not_exists:
            ensure_table(conn, table_name, create_sql)

        df = load_csv_to_df(csv_path)
        # Bulk load under the specified database/schema context
        success, nchunks, nrows, _ = write_pandas(
            conn=conn,
            df=df,
            table_name=table_name,
            database=cfg.get("database"),
            schema=cfg.get("schema"),
            quote_identifiers=False  # our column names are simple
        )
        print(f"write_pandas success={success}, chunks={nchunks}, rows={nrows}")
        if not success:
            raise RuntimeError("Snowflake write_pandas failed")
        return nrows
    finally:
        conn.close()
