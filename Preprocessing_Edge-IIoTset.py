
from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

logger = logging.getLogger("preprocessing")

DROP_COLUMNS = [
    "frame.time", "ip.src_host", "ip.dst_host", "arp.dst.proto_ipv4", "arp.src.proto_ipv4",
    "http.file_data", "http.request.uri.query",
    "http.request.full_uri", "tcp.options", "tcp.payload",
    "tcp.srcport", "mqtt.msg", "Attack_label",
]

RENAME_MAP = {
    "http.request.method": "http1", "http.referer": "http2", "http.request.version": "http3",
    "dns.qry.name.len": "dns", "mqtt.conack.flags": "mqtt1",
    "mqtt.protoname": "mqtt2", "mqtt.topic": "mqtt3",
}

ENCODE_COLUMNS = ["http1", "http2", "http3", "dns", "mqtt1", "mqtt2", "mqtt3"]

ZERO_COLUMN_THRESHOLD = 99.0 

ATTACK_TYPE_COLLAPSE = [
    ("DDoS_TCP", "DDoS"), ("DDoS_UDP", "DDoS"), ("DDoS_HTTP", "DDoS"), ("DDoS_ICMP", "DDoS"),
    ("Port_Scanning", "Scanning"), ("Fingerprinting", "Scanning"), ("Vulnerability_scanner", "Scanning"),
    ("MITM", "MITM"),
    ("XSS", "Injection"), ("SQL_injection", "Injection"), ("Uploading", "Injection"),
    ("Backdoor", "Malware"), ("Password", "Malware"), ("Ransomware", "Malware"),
]

def load_raw(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    logger.info("Loaded raw: %s rows x %s cols from %s", f"{df.shape[0]:,}", df.shape[1], input_path)
    return df

def drop_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=DROP_COLUMNS)
    df = df.rename(columns=RENAME_MAP)
    return df

def encode_categoricals(df: pd.DataFrame, artifacts_dir: Path) -> pd.DataFrame:
    
    label_encoders: Dict[str, LabelEncoder] = {}
    for col in ENCODE_COLUMNS:
        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col])
        label_encoders[col] = le

    onehot_encoders: Dict[str, OneHotEncoder] = {}
    onehot_frames: List[pd.DataFrame] = [df]  
    for col in ENCODE_COLUMNS:
        ohe = OneHotEncoder()
        X = ohe.fit_transform(df[f"{col}_encoded"].values.reshape(-1, 1)).toarray()
        onehot_encoders[col] = ohe
        frame = pd.DataFrame(
            X, columns=[f"{col}_{int(i)}" for i in range(X.shape[1])], index=df.index
        )
        onehot_frames.append(frame)

    df = pd.concat(onehot_frames, axis=1)
    df = df.drop(columns=ENCODE_COLUMNS)

    with open(artifacts_dir / "label_encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)
    with open(artifacts_dir / "onehot_encoders.pkl", "wb") as f:
        pickle.dump(onehot_encoders, f)
    logger.info("Encoders fit on columns %s, pickled to %s", ENCODE_COLUMNS, artifacts_dir)

    return df

def drop_sparse_columns(df: pd.DataFrame) -> pd.DataFrame:
    percentage_zeros = (df == 0).mean() * 100
    columns_to_drop = percentage_zeros[percentage_zeros > ZERO_COLUMN_THRESHOLD].index.tolist()
    df = df.drop(columns=columns_to_drop)
    logger.info("Dropped %d columns > %.0f%% zero-valued: %s",
                len(columns_to_drop), ZERO_COLUMN_THRESHOLD, columns_to_drop)
    return df

def collapse_attack_types(df: pd.DataFrame) -> pd.DataFrame:
    for src, dst in ATTACK_TYPE_COLLAPSE:
        df["Attack_type"] = df["Attack_type"].str.replace(src, dst, regex=False)
    return df

def write_schema(df: pd.DataFrame, output_path: str, input_path: str, artifacts_dir: Path) -> dict:
    feature_columns = [c for c in df.columns if c != "Attack_type"]
    schema = {
        "source_raw_file": str(input_path),
        "output_file": str(output_path),
        "preprocessed_at_unix": time.time(),
        "row_count": int(len(df)),
        "n_feature_columns": len(feature_columns),
        "feature_columns": feature_columns,  # order matters: must match ids_pipeline's feature_cols
        "attack_type_value_counts": df["Attack_type"].value_counts().to_dict(),
    }
    schema_path = artifacts_dir / "feature_schema.json"
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    logger.info("Schema written to %s (%d rows, %d feature columns)",
                schema_path, schema["row_count"], schema["n_feature_columns"])
    return schema

def run_preprocessing(input_path: str, output_path: str, artifacts_dir: str = "./artifacts") -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw(input_path)
    df = drop_and_rename(df)
    df = encode_categoricals(df, out_dir)
    df = drop_sparse_columns(df)
    df = collapse_attack_types(df)

    logger.info("Attack_type distribution:\n%s", df["Attack_type"].value_counts().to_string())

    df.to_csv(output_path, index=False)
    logger.info("Wrote %s rows x %s cols to %s", f"{df.shape[0]:,}", df.shape[1], output_path)

    schema = write_schema(df, output_path, input_path, out_dir)
    return schema

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="DNN-EdgeIIoT-dataset.csv")
    parser.add_argument("--output", default="Edge-IIoTset.csv")
    parser.add_argument("--artifacts-dir", default="./artifacts")
    args = parser.parse_args()
    run_preprocessing(args.input, args.output, args.artifacts_dir)