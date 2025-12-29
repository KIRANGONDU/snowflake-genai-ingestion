import os
import pandas as pd
 
from target_table_schema import CANONICAL_SCHEMA
from ai_column_mapping.column_mapper_factory import get_column_mapper
from transformations import split_fullname, standardize_dob, add_load_date
 
INPUT_DIR = "source_data"
OUTPUT_DIR = "staging_data"
OUTPUT_FILE = "Transformed_file.csv"
 
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
mapper = get_column_mapper()
final_dfs = []
 
for file in os.listdir(INPUT_DIR):
    path = os.path.join(INPUT_DIR, file)
 
    if file.lower().endswith(".csv"):
        df = pd.read_csv(path)
    elif file.lower().endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        continue
 
    df.columns = [c.strip() for c in df.columns]
    mapping = mapper.map_columns(df.columns.tolist(), CANONICAL_SCHEMA)
 
    print(f"\n {file}")
    for k, v in mapping.items():
        print(f"  {k} -> {v}")
 
    std_df = pd.DataFrame(index=df.index)
 
    for src, tgt in mapping.items():
        if tgt and tgt not in {"firstname", "lastname", "fullname"}:
            std_df[tgt] = df[src]
 
    if "firstname" in mapping.values():
        col = next(k for k, v in mapping.items() if v == "firstname")
        std_df["firstname"] = df[col]
 
    if "lastname" in mapping.values():
        col = next(k for k, v in mapping.items() if v == "lastname")
        std_df["lastname"] = df[col]
 
    if "fullname" in mapping.values() and (
        "firstname" not in std_df.columns or std_df["firstname"].isna().all()
    ):
        col = next(k for k, v in mapping.items() if v == "fullname")
        fn, ln = zip(*df[col].apply(split_fullname))
        std_df["firstname"] = fn
        std_df["lastname"] = ln
 
    if "dob" in std_df.columns:
        std_df["dob"] = std_df["dob"].apply(standardize_dob)
    else:
        std_df["dob"] = None
 
    for col in CANONICAL_SCHEMA:
        if col not in std_df:
            std_df[col] = None
 
    std_df["customer_id"] = std_df["customer_id"].fillna(
        std_df["email"].apply(lambda x: abs(hash(x)) % 10**9 if pd.notna(x) else None)
    )
 
    std_df = add_load_date(std_df)
    std_df = std_df[list(CANONICAL_SCHEMA.keys())]
 
    final_dfs.append(std_df)
 
final_df = pd.concat(final_dfs, ignore_index=True)
out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
final_df.to_csv(out_path, index=False)
 
print(f"\n FINAL OUTPUT CREATED: {out_path}")