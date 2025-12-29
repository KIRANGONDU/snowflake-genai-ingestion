import pandas as pd
from datetime import datetime


def split_fullname(fullname):
    if fullname is None or pd.isna(fullname):
        return None, None

    fullname = fullname.replace(",", " ").strip()
    parts = fullname.split()

    firstname = parts[0] if parts else None
    lastname = " ".join(parts[1:]) if len(parts) > 1 else None

    return firstname, lastname


def standardize_dob(dob_value):
    if dob_value is None or pd.isna(dob_value):
        return None

    dob_value = str(dob_value).strip()

    formats = ["%d-%m-%Y", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"]

    for fmt in formats:
        try:
            return pd.to_datetime(dob_value, format=fmt).strftime("%Y%m%d")
        except ValueError:
            continue

    try:
        return pd.to_datetime(dob_value, errors="coerce").strftime("%Y%m%d")
    except Exception:
        return None


def add_load_date(df):
    df["load_date"] = datetime.today().strftime("%Y-%m-%d")
    return df
