import hashlib
import numpy as np
import pandas as pd

REQUIRED_COLS = [
    "record_id", "plot_area_ha", "rainfall_mm", "soil_ph", "seed_kg",
    "distance_km", "arrival_hour", "actual_yield_kg", "dispatch_attention"
]
FEATURES = ["plot_area_ha", "rainfall_mm", "soil_ph",
            "seed_kg", "distance_km", "arrival_hour"]
TARGET_REG = "actual_yield_kg"
TARGET_CLF = "dispatch_attention"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_validate(path):
    df = pd.read_csv(path)
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    for c in REQUIRED_COLS:
        if c == "record_id":
            continue
        if not np.issubdtype(df[c].dtype, np.number):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    missing = df.isna().sum().to_dict()
    dup = int(df.duplicated(subset=["record_id"]).sum())

    df = df.dropna(subset=FEATURES + [TARGET_REG, TARGET_CLF]).reset_index(drop=True)
    return df, missing, dup


def build_feature_matrix(df):
    return df[FEATURES].to_numpy(dtype=float)


def build_report(df, missing, dup, group, sha):
    desc = df[FEATURES + [TARGET_REG]].describe().to_dict()
    return {
        "group_code": group,
        "sha256": sha,
        "row_count": int(len(df)),
        "feature_count": len(FEATURES),
        "missing_values": missing,
        "duplicate_record_ids": dup,
        "descriptive_stats": desc,
        "features": FEATURES,
    }