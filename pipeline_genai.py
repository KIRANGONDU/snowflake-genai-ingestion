
import os
import json
import hashlib
import re
import time
import csv
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

# =========================
# Config & Canonical Schema
# =========================
CANONICAL_SCHEMA = {
    "customer_id": "string",
    "firstname": "string",
    "lastname": "string",
    "gender": "string",
    "dob": "date",
    "email": "string",
    "phone": "string",
    "city": "string",
    "country": "string",
    "load_date": "date",
}

INPUT_DIR = "source_data"
OUTPUT_DIR = "staging_files"
OUTPUT_FILE = "customers_standardized.csv"

USE_REAL_GENAI = True
LLM_PROVIDER = "gemini"
LLM_MODEL = "gemini-2.5-flash"

# ⛔️ If you must manually add the key, put it here:
GEMINI_API_KEY = "AIzaSyCLFip_m-9y_cdUcAowjAY1-dtj1fpXXbM"

CACHE_DIR = ".llm_cache"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# =========================
# Cache helpers
# =========================
def _cache_key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def cache_get(namespace: str, payload: dict) -> Optional[dict]:
    key = _cache_key(payload)
    path = os.path.join(CACHE_DIR, f"{namespace}_{key}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

def cache_put(namespace: str, payload: dict, result: dict):
    key = _cache_key(payload)
    path = os.path.join(CACHE_DIR, f"{namespace}_{key}.json")
    with open(path, "w") as f:
        json.dump(result, f)

# =========================
# Field transformations
# =========================
def standardize_dob(x):
    if pd.isna(x): return None
    s = str(x).strip()
    if not s: return None
    # YYYYMMDD
    if re.fullmatch(r"\d{8}", s):
        try:
            return datetime.strptime(s, "%Y%m%d").date().isoformat()
        except:
            return None
    # Common formats
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except:
            continue
    # Pandas last resort
    try:
        return pd.to_datetime(s, errors="coerce").date().isoformat()
    except:
        return None

def normalize_gender(x):
    if pd.isna(x): return None
    s = str(x).strip().lower()
    if s in {"m", "male"}: return "Male"
    if s in {"f", "female"}: return "Female"
    return None

def split_fullname(fullname):
    if pd.isna(fullname): return (None, None)
    s = str(fullname).strip()
    if not s: return (None, None)
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) == 1: return (parts[0], None)
        return (parts[0], " ".join(parts[1:]))
    parts = [p for p in re.split(r"\s+", s) if p]
    if len(parts) == 1: return (parts[0], None)
    return (parts[0], " ".join(parts[1:]))

def name_from_email(email):
    if pd.isna(email): return (None, None)
    s = str(email).strip()
    if "@" not in s: return (None, None)
    local = s.split("@")[0]
    tokens = [t for t in re.split(r"[._\-]+", local) if t]
    if not tokens: return (None, None)
    tokens = [t.capitalize() for t in tokens]
    if len(tokens) == 1: return (tokens[0], None)
    return (tokens[0], " ".join(tokens[1:]))

def build_name(firstname, lastname):
    if firstname and lastname: return f"{firstname} {lastname}"
    if firstname: return firstname
    if lastname: return lastname
    return "Unknown"

def titlecase_or_none(x):
    if pd.isna(x): return None
    s = str(x).strip()
    return s.title() if s else None

# =========================
# LLM call helpers
# =========================
def _parse_json_strict(text: str) -> dict:
    """Strict JSON parse; extract the first {...} block if needed."""
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            return json.loads(candidate)
        raise

def call_llm_json(system_prompt: str, user_prompt: str) -> dict:
    """Invoke Gemini; return parsed JSON; retries included."""
    if not USE_REAL_GENAI:
        raise RuntimeError("LLM disabled by configuration.")
    if LLM_PROVIDER.lower() != "gemini":
        raise RuntimeError("This build supports only Gemini for direct key usage.")
    if not GEMINI_API_KEY or GEMINI_API_KEY.strip() == "" or "PASTE_" in GEMINI_API_KEY:
        raise RuntimeError("Gemini API key missing. Set GEMINI_API_KEY near the top of the file.")

    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)

    generation_config = {
        "response_mime_type": "application/json",
        "temperature": 0.0,
    }

    last_err = None
    for attempt in range(3):
        try:
            model = genai.GenerativeModel(
                LLM_MODEL,
                system_instruction=system_prompt
            )
            resp = model.generate_content(
                user_prompt,
                generation_config=generation_config,
            )

            text = getattr(resp, "text", None)
            if not text:
                try:
                    text = resp.candidates[0].content.parts[0].text
                except Exception:
                    raise ValueError("Gemini response had no text payload.")

            return _parse_json_strict(text)
        except Exception as e:
            last_err = e
            time.sleep(0.8 * (attempt + 1))

    raise RuntimeError(f"LLM call failed after retries: {last_err}")

# =========================
# Prompt templates
# =========================
SYSTEM_MAP = (
    "You are a data engineering assistant that maps source table columns to a canonical schema. "
    "Return ONLY valid JSON with keys of the source columns and values: one of the canonical field names or null. "
    "Do NOT invent fields, do NOT hallucinate. Prefer semantic meaning. If composite full name, map to 'name'. "
    "If multiple source columns map to the same target, choose the best and set the others to null. Output must be pure JSON."
)

def USER_MAP(source_cols: List[str]) -> str:
    allowed = list(CANONICAL_SCHEMA.keys())
    examples = (
        '["cust_id","first_name","last_name","sex","birthdate","mail","mobile","city","country","ingest_date"]\n'
        '→ {"cust_id":"customer_id","first_name":"firstname","last_name":"lastname","sex":"gender","birthdate":"dob",'
        '"mail":"email","mobile":"phone","city":"city","country":"country","ingest_date":"load_date"}\n\n'
        '["id","fullname","gender","dob","email","phone","load_date"]\n'
        '→ {"id":"customer_id","fullname":"name","gender":"gender","dob":"dob","email":"email","phone":"phone","load_date":"load_date"}'
    )
    return (
        f"Canonical schema fields (allowed targets): {allowed}\n\n"
        f"Source columns:\n{source_cols}\n\n"
        f"Examples:\n{examples}\n\n"
        f"Return ONLY JSON for your mapping for the provided source columns."
    )

SYSTEM_NAME = (
    'Split or infer firstname/lastname. Return ONLY JSON: {"firstname": "...", "lastname": "..."}; unknown → null.'
)
def USER_NAME(fullname, firstname, lastname, email) -> str:
    return (
        f'Input: fullname="{fullname}", firstname="{firstname}", lastname="{lastname}", email="{email}"\n'
        'Examples:\n- fullname="Siva,Kumar" → {"firstname":"Siva","lastname":"Kumar"}\n'
        '- fullname="Anil Raj" → {"firstname":"Anil","lastname":"Raj"}\n'
        '- email="priya@gmail.com" → {"firstname":"Priya","lastname":null}\n'
        'Return ONLY JSON.'
    )

SYSTEM_GENDER = 'Normalize gender to "Male" or "Female" when possible; else null. Return ONLY JSON: {"gender":"..."}'
def USER_GENDER(gender) -> str:
    return f'Input: gender="{gender}"\nExamples: "M"→Male, "F"→Female, "male"→Male.\nReturn ONLY JSON.'

SYSTEM_DOB = 'Convert date to ISO YYYY-MM-DD; invalid → null. Return ONLY JSON: {"dob":"..."}'
def USER_DOB(dob) -> str:
    return f'Input: dob="{dob}"\nExamples: "19940512"→"1994-05-12"; "21/03/1995"→"1995-03-21".\nReturn ONLY JSON.'

# =========================
# Column mapping heuristics
# =========================
ALIASES = {
    # ---- ID variants -> customer_id ----
    "empid": "customer_id",
    "emp_id": "customer_id",
    "employee_id": "customer_id",
    "employeeid": "customer_id",
    "employee-no": "customer_id",
    "employee_no": "customer_id",
    "employee number": "customer_id",
    "cust_id": "customer_id",
    "customerid": "customer_id",
    "customer_id": "customer_id",
    "id": "customer_id",

    # ---- Name variants ----
    "first_name": "firstname",
    "fname": "firstname",
    "last_name": "lastname",
    "lname": "lastname",
    "surname": "lastname",
    "full_name": "name",
    "fullname": "name",
    "name": "name",

    # ---- Gender variants ----
    "sex": "gender",
    "gender": "gender",

    # ---- DOB variants ----
    "dob": "dob",
    "birthdate": "dob",
    "date_of_birth": "dob",
    "dateofbirth": "dob",

    # ---- Email variants ----
    "mail": "email",
    "email": "email",

    # ---- Phone variants ----
    "phone": "phone",
    "mobile": "phone",
    "mobile_no": "phone",
    "mobile number": "phone",
    "phone_number": "phone",

    # ---- City variants ----
    "city": "city",
    "location": "city",
    "place": "city",
    "town": "city",

    # ---- Country variants ----
    "country": "country",

    # ---- Load date variants ----
    "load_date": "load_date",
    "ingest_date": "load_date",
}

ID_KEYWORDS = {"emp", "employee", "cust", "customer"}

def guess_customer_id_target(colname: str) -> bool:
    """
    Heuristic: map any 'id'-like columns to customer_id
    if they include (emp|employee|cust|customer) + id, or are exactly 'id'.
    """
    lc = colname.lower().strip()
    # normalize separators
    norm = re.sub(r"[^a-z0-9]+", " ", lc)
    tokens = set(norm.split())

    # exact id
    if lc == "id":
        return True

    # contains id and any of the id keywords
    if "id" in tokens or "id" in lc:
        if any(k in tokens or k in lc for k in ID_KEYWORDS):
            return True

    # common patterns like EMPID, EMP_ID, EMP-ID, EMP.ID
    if re.fullmatch(r"(emp|employee|cust|customer)[\s_\-]*id", norm):
        return True

    # endings like ..._id, employeeid
    if lc.endswith("_id") or lc.endswith("id"):
        if any(k in lc for k in ID_KEYWORDS):
            return True

    return False

def enforce_unique_targets(mapping: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """
    If multiple source columns mapped to the same target, keep the best one:
    - Prefer exact canonical match (src name equals target exactly).
    - Then prefer alias matches.
    - Others become None.
    """
    canonical_set = set(CANONICAL_SCHEMA.keys())
    target_to_sources: Dict[str, List[str]] = {}
    for src, tgt in mapping.items():
        if tgt in canonical_set:
            target_to_sources.setdefault(tgt, []).append(src)

    for tgt, sources in target_to_sources.items():
        if len(sources) <= 1:
            continue
        # Rank sources: exact match first, then alias
        exact = [s for s in sources if s.strip().lower() == tgt]
        alias = [s for s in sources if ALIASES.get(s.strip().lower(), None) == tgt]
        keep = exact[0] if exact else (alias[0] if alias else sources[0])
        for s in sources:
            if s != keep:
                mapping[s] = None
    return mapping

def map_columns_with_llm(src_cols: List[str]) -> Dict[str, Optional[str]]:
    # Clean and normalize headers
    src_cols_clean = [str(c).strip() for c in src_cols]
    canonical_set = set(CANONICAL_SCHEMA.keys())

    # 1) Deterministic pre-mapping + heuristics
    mapping: Dict[str, Optional[str]] = {}
    for c in src_cols_clean:
        lc = c.lower().strip()
        if lc in canonical_set:
            mapping[c] = lc
            continue
        if lc in ALIASES:
            mapping[c] = ALIASES[lc]
            continue
        # Heuristic for ID-like columns
        if guess_customer_id_target(c):
            mapping[c] = "customer_id"
            continue
        mapping[c] = None

    # 2) Ask LLM for unresolved columns only
    unresolved = [c for c in src_cols_clean if mapping[c] is None]
    if unresolved:
        payload = {"src_cols": unresolved, "schema": list(CANONICAL_SCHEMA.keys())}
        cached = cache_get("column_map", payload)
        if cached:
            for c in unresolved:
                tgt = cached.get(c, None)
                mapping[c] = tgt if tgt in canonical_set else None
        else:
            try:
                result = call_llm_json(SYSTEM_MAP, USER_MAP(unresolved))
                for c in unresolved:
                    tgt = result.get(c, None)
                    mapping[c] = tgt if tgt in canonical_set else None
                cache_put("column_map", payload, {c: mapping[c] for c in unresolved})
            except Exception:
                # Leave unresolved as None
                pass

    # 3) Ensure a single source per target
    mapping = enforce_unique_targets(mapping)
    return mapping

# =========================
# Optional LLM-assisted transformations (fill-only)
# =========================
def llm_name_synthesis(fullname, firstname, lastname, email):
    try:
        res = call_llm_json(SYSTEM_NAME, USER_NAME(fullname, firstname, lastname, email))
        return res.get("firstname"), res.get("lastname")
    except:
        # Fallback
        if (firstname and str(firstname).strip()) or (lastname and str(lastname).strip()):
            return (firstname if firstname else None, lastname if lastname else None)
        if fullname and str(fullname).strip():
            return split_fullname(fullname)
        return name_from_email(email)

def llm_gender_normalize(gender):
    try:
        res = call_llm_json(SYSTEM_GENDER, USER_GENDER(gender))
        g = res.get("gender")
        return g if g in {"Male", "Female"} else None
    except:
        return normalize_gender(gender)

def llm_dob_standardize(dob):
    try:
        res = call_llm_json(SYSTEM_DOB, USER_DOB(dob))
        dob_iso = res.get("dob")
        return dob_iso if dob_iso else standardize_dob(dob)
    except:
        return standardize_dob(dob)

# =========================
# Per-file processing
# =========================
def process_one_file(file_path: str) -> pd.DataFrame:
    # Robust CSV/Excel reading
    if file_path.lower().endswith(".csv"):
        df = pd.read_csv(
            file_path,
            dtype=str,
            encoding="utf-8-sig",
            keep_default_na=True,
            na_values=["", "NULL", "null"],
            quoting=csv.QUOTE_MINIMAL,
            quotechar='"',
            escapechar='\\'
        )
    else:
        df = pd.read_excel(file_path, dtype=str, engine="openpyxl")

    # Clean headers
    df.columns = [str(c).strip() for c in df.columns]
    src_cols = df.columns.tolist()

    # Map columns
    mapping = map_columns_with_llm(src_cols)

    # Diagnostics
    print(f"\nFile: {os.path.basename(file_path)}")
    print("Source columns:", src_cols)
    print("Mapping:")
    for src, tgt in mapping.items():
        print(f"  {src} -> {tgt}")

    # Start standardized DF
    std_df = pd.DataFrame()

    # Copy mapped columns (ignore non-canonical; skip 'name' — we'll build it)
    for src_col, tgt_col in mapping.items():
        if tgt_col and tgt_col in CANONICAL_SCHEMA and tgt_col != "name":
            if src_col in df.columns:
                std_df[tgt_col] = df[src_col]

    # Title-case city for consistency
    if "city" in std_df.columns:
        std_df["city"] = std_df["city"].apply(titlecase_or_none)

    # Gather raw fullname/name for synthesis
    raw_fullname = None
    src_name_col = next((k for k, v in mapping.items() if v == "name"), None)
    if src_name_col:
        raw_fullname = df[src_name_col]
    elif "fullname" in df.columns:
        raw_fullname = df["fullname"]

    # Fill-only synthesis for firstname/lastname
    first_list, last_list = [], []
    n_rows = len(df)

    for i in range(n_rows):
        fn = std_df["firstname"].iloc[i] if "firstname" in std_df.columns else None
        ln = std_df["lastname"].iloc[i] if "lastname" in std_df.columns else None
        em = std_df["email"].iloc[i] if "email" in std_df.columns else None
        full = raw_fullname.iloc[i] if raw_fullname is not None else None

        if (fn and str(fn).strip()) or (ln and str(ln).strip()):
            # Keep existing values
            first_list.append(fn)
            last_list.append(ln)
        else:
            f, l = llm_name_synthesis(full, fn, ln, em)
            first_list.append(f)
            last_list.append(l)

    # Apply fills
    if "firstname" not in std_df.columns:
        std_df["firstname"] = pd.Series(first_list)
    else:
        std_df["firstname"] = std_df["firstname"].where(
            std_df["firstname"].notna() & (std_df["firstname"].astype(str).str.strip() != ""),
            pd.Series(first_list)
        )

    if "lastname" not in std_df.columns:
        std_df["lastname"] = pd.Series(last_list)
    else:
        std_df["lastname"] = std_df["lastname"].where(
            std_df["lastname"].notna() & (std_df["lastname"].astype(str).str.strip() != ""),
            pd.Series(last_list)
        )

    # Build 'name' from first+last
    std_df["name"] = [build_name(f, l) for f, l in zip(std_df["firstname"], std_df["lastname"])]

    # Normalize gender & dob (preserve good values)
    if "gender" in std_df.columns:
        std_df["gender"] = std_df["gender"].apply(
            lambda x: llm_gender_normalize(x) if pd.notna(x) and str(x).strip() else None
        )
    if "dob" in std_df.columns:
        std_df["dob"] = std_df["dob"].apply(
            lambda x: llm_dob_standardize(x) if pd.notna(x) and str(x).strip() else None
        )

    # Ensure all canonical columns exist; ignore non-canonical
    for col in CANONICAL_SCHEMA:
        if col not in std_df.columns:
            std_df[col] = None

    # Load date handling
    if std_df["load_date"].isna().all():
        std_df["load_date"] = pd.Timestamp("today").date().isoformat()
    else:
        std_df["load_date"] = std_df["load_date"].apply(
            lambda x: pd.to_datetime(x, errors="coerce").date().isoformat()
            if pd.notna(x) else pd.Timestamp("today").date().isoformat()
        )

    # Final order (canonical only)
    std_df = std_df[list(CANONICAL_SCHEMA.keys())]

    # Optional quick preview
    print("\nStandardized preview (first 3 rows):")
    print(std_df.head(3).to_string(index=False))
    return std_df

# =========================
# Entry point
# =========================
def main_run() -> str:
    """
    Run the GENAI pipeline using current module-level settings:
      INPUT_DIR, OUTPUT_DIR, OUTPUT_FILE, USE_REAL_GENAI, LLM_PROVIDER, LLM_MODEL, GEMINI_API_KEY
    Returns: output CSV file path.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    final_df_list: List[pd.DataFrame] = []

    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".csv", ".xlsx"))]
    print("Detected files:", files)
    if not files:
        raise RuntimeError(f"No input files found in '{INPUT_DIR}' matching .csv/.xlsx")

    for file in files:
        file_path = os.path.join(INPUT_DIR, file)
        try:
            std_df = process_one_file(file_path)
            final_df_list.append(std_df)
        except Exception as e:
            print(f"[warn] Skipping '{file}' due to error: {e}")

    if not final_df_list:
        raise RuntimeError("No files processed successfully.")

    final_df = pd.concat(final_df_list, ignore_index=True)
    final_df = final_df.drop_duplicates()  # optional de-dup
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    final_df.to_csv(out_path, index=False)
    print(f"\nStandardized data saved successfully → {out_path}")
    return out_path

if __name__ == "__main__":
    main_run()
