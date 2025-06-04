import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from Bio import SeqIO
import subprocess
import re
import json

def parse_fasta(fasta_path):
    records = list(SeqIO.parse(fasta_path, "fasta"))

    def extract_start_stop(label):
        match = re.search(r'-(\d+)-(\d+)$', label)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None

    labels = [rec.id.split('|')[1].replace("label=", "", 1) for rec in records]
    starts, stops = zip(*(extract_start_stop(label) for label in labels))

    return pd.DataFrame({
        "AC": [rec.id.split('|')[0] for rec in records],
        "label": labels,
        "start": starts,
        "end": stops,
        "length": [len(rec.seq) for rec in records],
        "header": [rec.id for rec in records],
        "sequence": [str(rec.seq) for rec in records]
    })

def write_fasta(df, path):
    with open(path, "w") as f:
        for _, row in df.iterrows():
            f.write(f">{row['header']}\n{row['sequence']}\n")

def main(fasta_path, out_dir, train_size, val_size, test_size, mmseqs_threshold, cov=0.8, cov_mode=2):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fasta_path_base = Path(fasta_path).name
    out_dir = out_dir / fasta_path_base.replace('.fasta', '')
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    fasta_df = parse_fasta(fasta_path)
    n_total = len(fasta_df)

    # 1. MMseqs2 easy-cluster for deduplication
    mmseqs_dir = out_dir / "clustered_results"
    mmseqs_dir.mkdir(parents=True, exist_ok=True)
    input_fasta = mmseqs_dir / "input.fasta"
    write_fasta(fasta_df, input_fasta)

    # Run easy-cluster
    subprocess.run([
        "mmseqs", "easy-cluster", str(input_fasta), str(mmseqs_dir) , str(out_dir / "tmp"),
        "--min-seq-id", str(mmseqs_threshold),
        "-c", str(cov),
        "--cov-mode", str(cov_mode)
    ], check=True)

    # The deduplicated fasta is at mmseqs_dir/rep_seq.fasta
    rep_fasta = mmseqs_dir / "rep_seq.fasta"
    dedup_df = parse_fasta(rep_fasta)
    dedup_df.to_csv(out_dir / "dedup_sequences.csv", index=False)
    n_dedup = len(dedup_df)
    print(f"Deduplicated: {n_dedup} sequences remain ({n_dedup/n_total:.2%} of original, {100 - (n_dedup/n_total)*100:.2f}% lost)")

    # Count clusters (number of lines in cluster.tsv)
    cluster_file = mmseqs_dir / "cluster.tsv"
    with open(cluster_file) as f:
        n_clusters = sum(1 for _ in f)
    print(f"Number of clusters: {n_clusters}")

    # 2. Split into train/val/test
    total = train_size + val_size + test_size
    train_ratio = train_size / total
    val_ratio = val_size / total
    test_ratio = test_size / total

    train_df, temp_df = train_test_split(dedup_df, test_size=(1-train_ratio), random_state=42)
    val_relative = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(temp_df, test_size=(1-val_relative), random_state=42)

    # Save to CSV
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    print(f"Final split sizes: train={len(train_df)} ({len(train_df)/n_dedup:.2%}), val={len(val_df)} ({len(val_df)/n_dedup:.2%}), test={len(test_df)} ({len(test_df)/n_dedup:.2%})")

    # Save meta data
    meta = {
        "n_total": n_total,
        "n_dedup": n_dedup,
        "n_clusters": n_clusters,
        "percent_lost": 100 - (n_dedup / n_total) * 100,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "train_ratio": len(train_df) / n_dedup,
        "val_ratio": len(val_df) / n_dedup,
        "test_ratio": len(test_df) / n_dedup,
        "mmseqs_min_seq_id": mmseqs_threshold,
        "mmseqs_cov": cov,
        "mmseqs_cov_mode": cov_mode
    }
    with open(out_dir / "meta_data.json", "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split FASTA with MMseqs2 easy-cluster deduplication and train/val/test split.")
    parser.add_argument("--fasta", required=True, help="Path to input FASTA file")
    parser.add_argument("--out_dir", required=True, help="Output directory for split CSVs")
    parser.add_argument("--train_size", type=float, default=0.7, help="Proportion for train set (default: 0.7)")
    parser.add_argument("--val_size", type=float, default=0.2, help="Proportion for val set (default: 0.2)")
    parser.add_argument("--test_size", type=float, default=0.1, help="Proportion for test set (default: 0.1)")
    parser.add_argument("--mmseqs_threshold", type=float, default=0.3, help="MMseqs2 min-seq-id threshold (default: 0.3)")
    parser.add_argument("--cov", type=float, default=0.8, help="MMseqs2 coverage threshold (default: 0.8)")
    parser.add_argument("--cov_mode", type=int, default=0, help="MMseqs2 coverage mode (default: 0)")
    args = parser.parse_args()
    main(args.fasta, args.out_dir, args.train_size, args.val_size, args.test_size, args.mmseqs_threshold, args.cov, args.cov_mode)