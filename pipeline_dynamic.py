
import os
import re
import csv
import json
import time
import hashlib
import pandas as pd
from typing import List, Dict, Optional

# =========================
# IO
# =========================
INPUT_DIR = "flight_data"
OUTPUT_DIR = "fligth_output"
OUTPUT_FILE = "flights_standardized.csv"

# =========================
# Config (LLM optional; keep off for strict mapping)
# =========================
USE_REAL_GENAI = False                # keep False to avoid any LLM-induced shuffling
LLM_PROVIDER = "gemini"
LLM_MODEL = "gemini-1.5-flash"
GEMINI_API_KEY = "AIzaSyCLFip_m-9y_cdUcAowjAY1-dtj1fpXXbM"

CACHE_DIR = ".llm_cache"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# =========================
# Target schema (fixed 10 columns)
# =========================
FLIGHT_TARGET = [
    "flight_id","airline","source_city","departure_time","stops",
    "arrival_time","destination_city","duration","days_left","price"
]

# =========================
# Ranked aliases (strongest first for each target)
# =========================
RANKED_ALIASES = {
    "flight_id": ["flight_id", "flight", "id"],
    "airline": ["airline", "carrier"],
    "source_city": ["source_city", "src_city", "incoming_city", "from_city"],
    "departure_time": ["departure_time", "target_time", "takeoff_time"],
    "stops": ["stops", "layovers"],
    "arrival_time": ["arrival_time", "arrive_time", "landing_time"],
    "destination_city": ["destination_city", "dest_city", "to_city"],
    "duration": ["duration", "flight_time"],
    "days_left": ["days_left", "days_to_departure"],
    "price": ["price", "amount", "fare", "cost"],
}

def snake(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(s).strip()).strip("_").lower()
    return s or "col"

def titlecase_or_none(x):
    if pd.isna(x): return None
    s = str(x).strip()
    return s.title() if s else None

# Build a reverse alias lookup: header_token -> candidate target(s) with rank
REV_ALIASES: Dict[str, List[tuple]] = {}
for tgt, variants in RANKED_ALIASES.items():
    for rank, hdr in enumerate(variants):
        REV_ALIASES.setdefault(hdr, []).append((tgt, rank))

def map_headers_strict(src_cols: List[str]) -> Dict[str, Optional[str]]:
    """
    Per-file, strict mapping:
      - For each source header, try to match a target via ranked alias tokens.
      - Enforce single source per target: keep the lowest-rank match.
      - Unmapped -> None.
    """
    # Prepare candidate matches: target -> (src, rank)
    best_for_target: Dict[str, tuple] = {}
    # We also keep per-source chosen target
    mapping: Dict[str, Optional[str]] = {c: None for c in src_cols}

    for c in src_cols:
        token = snake(c)
        # Try exact token first; else see if token matches any alias key directly
        candidates = []
        # direct exact
        if token in REV_ALIASES:
            candidates.extend(REV_ALIASES[token])
        # no direct hit? try relaxed checks (e.g., 'index' should map to nothing)
        # We DO NOT do fuzzy beyond aliases to avoid shuffling.

        # Evaluate candidates: choose the best-ranked available target
        for tgt, rank in candidates:
            # If target already taken, compare ranks and keep the best (lowest rank wins)
            prev = best_for_target.get(tgt)
            if prev is None or rank < prev[1]:
                best_for_target[tgt] = (c, rank)

    # Build mapping respecting one source per target
    for tgt in FLIGHT_TARGET:
        if tgt in best_for_target:
            src = best_for_target[tgt][0]
            mapping[src] = tgt

    # All other sources remain unmapped (None)
    return mapping

# =========================
# Optional LLM (kept off by default)
# =========================
def _cache_key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def cache_get(namespace: str, payload: dict) -> Optional[dict]:
    path = os.path.join(CACHE_DIR, f"{namespace}_{_cache_key(payload)}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

def cache_put(namespace: str, payload: dict, result: dict):
    path = os.path.join(CACHE_DIR, f"{namespace}_{_cache_key(payload)}.json")
    with open(path, "w") as f:
        json.dump(result, f)

def _parse_json_strict(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{"); end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
        raise

def call_llm_json(system_prompt: str, user_prompt: str) -> dict:
    if not USE_REAL_GENAI:
        raise RuntimeError("LLM disabled by configuration.")
    if LLM_PROVIDER.lower() != "gemini":
        raise RuntimeError("This build supports only Gemini for direct usage.")
    if not GEMINI_API_KEY or "PASTE_" in GEMINI_API_KEY:
        raise RuntimeError("Gemini API key missing. Set GEMINI_API_KEY.")

    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    generation_config = {"response_mime_type": "application/json", "temperature": 0.0}

    last_err = None
    for attempt in range(3):
        try:
            model = genai.GenerativeModel(LLM_MODEL, system_instruction=system_prompt)
            resp = model.generate_content(user_prompt, generation_config=generation_config)
            text = getattr(resp, "text", None) or resp.candidates[0].content.parts[0].text
            return _parse_json_strict(text)
        except Exception as e:
            last_err = e
            time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after retries: {last_err}")

SYSTEM_MAP = (
    "Map source headers to a fixed flight schema (10 columns). "
    f"Allowed targets: {FLIGHT_TARGET}. "
    "Return ONLY JSON: {source_header: target_or_null}. Prefer semantic meaning; avoid duplicates."
)
def USER_MAP(headers: List[str]) -> str:
    return f"Headers: {headers}\nReturn ONLY JSON for these headers."

# =========================
# Transform rows -> final 10 columns
# =========================
def to_flight_frame(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    # Build per-file frame with nulls for all targets first
    std = pd.DataFrame({col: [None]*len(df) for col in FLIGHT_TARGET})

    # Copy ONLY mapped columns from THIS file
    for src, tgt in mapping.items():
        if tgt in FLIGHT_TARGET and src in df.columns:
            std[tgt] = df[src]

    # Clean cities (optional)
    for city_col in ["source_city","destination_city"]:
        if city_col in std.columns:
            std[city_col] = std[city_col].apply(titlecase_or_none)

    # Keep fixed order
    return std[FLIGHT_TARGET]

# =========================
# Main
# =========================
def main():
    frames = []
    for file in os.listdir(INPUT_DIR):
        if not file.endswith((".csv",".xlsx")):
            continue
        path = os.path.join(INPUT_DIR, file)
        if file.endswith(".csv"):
            df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=True,
                             na_values=["","NULL","null"], quoting=csv.QUOTE_MINIMAL, quotechar='"', escapechar='\\')
        else:
            df = pd.read_excel(path, dtype=str, engine="openpyxl")
        df.columns = [c.strip() for c in df.columns]
        frames.append((file, df))

    if not frames:
        print("No input files found.")
        return

    std_all = []
    for file, df in frames:
        print(f"\nProcessing: {file}")
        print("Source headers:", list(df.columns))

        # 1) strict alias mapping per file
        mapping = map_headers_strict(list(df.columns))

        # 2) optional: LLM for unresolved ONLY (kept OFF by default)
        unresolved = [c for c, t in mapping.items() if t is None]
        if USE_REAL_GENAI and unresolved:
            payload = {"src_cols": unresolved, "targets": FLIGHT_TARGET}
            cached = cache_get("flight_map", payload)
            llm_map = cached
            if not llm_map:
                try:
                    llm_map = call_llm_json(SYSTEM_MAP, USER_MAP(unresolved))
                    cache_put("flight_map", payload, llm_map)
                except Exception:
                    llm_map = None
            if llm_map:
                # accept only canonical + non-duplicate suggestions
                assigned = set(t for t in mapping.values() if t)
                for c in unresolved:
                    t = llm_map.get(c, None)
                    if t in FLIGHT_TARGET and t not in assigned:
                        mapping[c] = t
                        assigned.add(t)

        print("Mapping:", mapping)  # diagnostics
        std = to_flight_frame(df, mapping)
        print("Preview:", std.head(3).to_string(index=False))
        std_all.append(std)

    final_df = pd.concat(std_all, ignore_index=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    final_df.to_csv(out_path, index=False)
    print(f"\n✅ Standardized data saved → {out_path}")

if __name__ == "__main__":
    main()
