
import os, csv, re, json, hashlib, pandas as pd
from datetime import datetime

def run_classic_pipeline(input_dir: str, output_dir: str, output_file: str) -> str:
    from target_table_schema import CANONICAL_SCHEMA
    from ai_column_mapping.column_mapper_factory import get_column_mapper
    from transformations import split_fullname, standardize_dob, add_load_date

    os.makedirs(output_dir, exist_ok=True)
    mapper = get_column_mapper()
    final_dfs = []

    for file in os.listdir(input_dir):
        path = os.path.join(input_dir, file)
        if file.lower().endswith(".csv"):
            df = pd.read_csv(path)
        elif file.lower().endswith(".xlsx"):
            df = pd.read_excel(path, engine="openpyxl")
        else:
            continue

        df.columns = [c.strip() for c in df.columns]
        mapping = mapper.map_columns(df.columns.tolist(), CANONICAL_SCHEMA)

        std_df = pd.DataFrame(index=df.index)
        for src, tgt in mapping.items():
            if tgt and tgt not in {"firstname", "lastname", "fullname"}:
                std_df[tgt] = df[src]

        # firstname/lastname handling (using your original helpers)
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
    out_path = os.path.join(output_dir, output_file)
    final_df.to_csv(out_path, index=False)
    return out_path


def run_genai_pipeline(input_dir: str, output_dir: str, output_file: str,
                       use_real_genai: bool = True,
                       llm_provider: str = "gemini",
                       llm_model: str = "gemini-1.5-flash",
                       api_key: str | None = None) -> str:
    # Import everything from pipeline_genai but inject configuration dynamically
    import importlib
    genai_mod = importlib.import_module("pipeline_genai")

    # Override configuration safely
    genai_mod.INPUT_DIR = input_dir
    genai_mod.OUTPUT_DIR = output_dir
    genai_mod.OUTPUT_FILE = output_file
    genai_mod.USE_REAL_GENAI = use_real_genai
    genai_mod.LLM_PROVIDER = llm_provider
    genai_mod.LLM_MODEL = llm_model
    if api_key:  # prefer env var if not passed
        genai_mod.GEMINI_API_KEY = api_key
    else:
        genai_mod.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

    # Execute main routine (reuse its loop)
    # Slightly adapt: just run its body by calling a function you’ll add (see below)
    return genai_mod.main_run()  # you’ll add this helper in pipeline_genai.py