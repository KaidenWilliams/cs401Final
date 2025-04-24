# Code for Step 2: Make manifests. Manifests link every input file to a class / label so the model can train

import os
import argparse
import boto3
import pandas as pd
from sklearn.model_selection import train_test_split

# Get Spectogram files
def scan_s3_specs(bucket_name, prefix, max_chunk=1000):
    """Build base_filename → list of S3 keys for chunks ≤ max_chunk."""
    mapping = {}
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=prefix):
        key = obj.key
        if not key.endswith(".npy"):
            continue
        filename = os.path.basename(key)
        parts = filename.split("_chunk_")
        if len(parts) != 2:
            continue
        base, chunk_str = parts
        try:
            chunk = int(chunk_str.split(".")[0])
            if chunk > max_chunk:
                continue
        except ValueError:
            continue
        mapping.setdefault(base, []).append(key)
    return mapping

# Build Manifests, map spectogram file to label/class from directory structure
def build_manifests(df, spec_mapping):
    """Explode & stratified‐split the DataFrame, return train_df, val_df."""
    df["primary_label"] = df["scientific_name"]
    def get_specs(fn):
        base = os.path.splitext(os.path.basename(fn))[0]
        return spec_mapping.get(base, [])
    df["s3_key_list"] = df["filename"].apply(get_specs)
    df = df.explode("s3_key_list").dropna(subset=["s3_key_list"])
    df = df.rename(columns={"s3_key_list": "s3_key"})
    counts = df["primary_label"].value_counts()
    valid = counts[counts >= 2].index.tolist()
    if valid:
        df_strat = df[df["primary_label"].isin(valid)]
        train_strat, val = train_test_split(
            df_strat,
            test_size=0.2,
            random_state=42,
            stratify=df_strat["primary_label"],
        )
        rest = df[~df["primary_label"].isin(valid)]
        train = pd.concat([train_strat, rest], ignore_index=True)
    else:
        train, val = train_test_split(df, test_size=0.2, random_state=42)
    return train, val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--specs-s3-prefix", type=str,
        default=os.environ.get("SM_CHANNEL_SPECS"),
        help="S3 URI prefix for spectrogram .npy files, e.g. s3://bucket/path/"
    )
    parser.add_argument(
        "--train-csv-s3-uri", type=str,
        default=os.environ.get("SM_CHANNEL_TRAINCSV"),
        help="S3 URI of train.csv"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR"),
        help="Local container path to write manifests"
    )
    parser.add_argument(
        "--max-chunk", type=int, default=1000,
        help="Skip any _chunk_ number above this"
    )
    args = parser.parse_args()

    if not args.specs_s3_prefix.startswith("s3://"):
        raise ValueError("specs-s3-prefix must be a full S3 URI")
    _, rest = args.specs_s3_prefix.split("s3://", 1)
    bucket_name, prefix = rest.split("/", 1)

    # Get spectogram files
    spec_map = scan_s3_specs(bucket_name, prefix, args.max_chunk)
    print(f"Found {sum(len(v) for v in spec_map.values())} spectrogram files")

    # Build Manifests
    df = pd.read_csv(args.train_csv_s3_uri)
    train_df, val_df = build_manifests(df, spec_map)
    print(f"Train rows: {len(train_df)}, Val rows: {len(val_df)}")

    # Save Manifests
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train_manifest.csv")
    val_path = os.path.join(args.output_dir, "val_manifest.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path,   index=False)
    print(f"Wrote {train_path}, {val_path}")


# Entry Point
if __name__ == "__main__":
    main()
