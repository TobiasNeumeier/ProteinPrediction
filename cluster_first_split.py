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
    mmseqs_dir = out_dir
    subprocess.run([
        "mmseqs", "easy-cluster", str(fasta_path), str(mmseqs_dir /  "output" ), str(mmseqs_dir /  "tmp" ),
        "--min-seq-id", str(mmseqs_threshold),
        "-c", str(cov),
        "--cov-mode", str(cov_mode)
    ], check=True)
    rep_fasta = mmseqs_dir / "output_rep_seq.fasta"
    dedup_df = parse_fasta(rep_fasta)
    dedup_df.to_csv(out_dir / "dedup_sequences.csv", index=False)
    n_dedup = len(dedup_df)
    print(f"Deduplicated: {n_dedup} sequences remain ({n_dedup/n_total:.2%} of original, {100 - (n_dedup/n_total)*100:.2f}% lost)")
    cluster_file = mmseqs_dir / "output_cluster.tsv"
    cluster_df = pd.read_csv(cluster_file, sep='\t', header=None, names=["representative", "member"])
    # Cluster-First Split: assign clusters to splits, then assign all members accordingly
    clusters = cluster_df.groupby("representative")["member"].apply(list).reset_index()
    clusters = clusters.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle clusters

    # Calculate target sizes (number of sequences, not clusters)
    total = train_size + val_size + test_size
    n_total_seqs = len(fasta_df)
    target_train = int(train_size / total * n_total_seqs)
    target_val = int(val_size / total * n_total_seqs)
    target_test = n_total_seqs - target_train - target_val

    # Assign clusters greedily to get as close as possible to target sizes
    split_assignments = []
    split_counts = {"train": 0, "val": 0, "test": 0}
    split_targets = {"train": target_train, "val": target_val, "test": target_test}
    split_order = ["train", "val", "test"]

    for _, row in clusters.iterrows():
        # Decide which split needs this cluster most (least over target)
        best_split = min(
            split_order,
            key=lambda split: (
                split_counts[split] + len(row["member"]))/split_targets[split]
                if split_counts[split] < split_targets[split] else float('inf')
        )
        # If all splits are full, assign to the one with the least overflow
        if best_split == "test" and split_counts["test"] >= split_targets["test"]:
            best_split = min(split_order, key=lambda split: split_counts[split] - split_targets[split])
        split_assignments.append((best_split, row["member"]))
        split_counts[best_split] += len(row["member"])

    # Collect members for each split
    split_members = {"train": [], "val": [], "test": []}
    for split, members in split_assignments:
        split_members[split].extend(members)

    train_df = fasta_df[fasta_df["header"].isin(split_members["train"])]
    val_df = fasta_df[fasta_df["header"].isin(split_members["val"])]
    test_df = fasta_df[fasta_df["header"].isin(split_members["test"])]

    # sort by sequence length (for later speedup)
    train_df = train_df.sort_values(by="length", ascending=False).reset_index(drop=True)
    val_df = val_df.sort_values(by="length", ascending=False).reset_index(drop=True)
    test_df = test_df.sort_values(by="length", ascending=False).reset_index(drop=True)

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    print(f"Final split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)})")
    meta = {
        "n_total": n_total,
        "n_dedup": n_dedup,
        "n_clusters": len(clusters),
        "percent_lost": 100 - (n_dedup / n_total) * 100,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "mmseqs_min_seq_id": mmseqs_threshold,
        "mmseqs_cov": cov,
        "mmseqs_cov_mode": cov_mode
    }
    with open(out_dir / "meta_data.json", "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster-First Split: Assign clusters to train/val/test.")
    parser.add_argument("--fasta", required=True, help="Path to input FASTA file")
    parser.add_argument("--out_dir", required=True, help="Output directory for split CSVs")
    parser.add_argument("--train_size", type=float, default=0.7, help="Proportion for train set (default: 0.7)")
    parser.add_argument("--val_size", type=float, default=0.2, help="Proportion for val set (default: 0.2)")
    parser.add_argument("--test_size", type=float, default=0.1, help="Proportion for test set (default: 0.1)")
    parser.add_argument("--mmseqs_threshold", type=float, default=0.3, help="MMseqs2 min-seq-id threshold (default: 0.3)")
    parser.add_argument("--cov", type=float, default=0.8, help="MMseqs2 coverage threshold (default: 0.8)")
    parser.add_argument("--cov_mode", type=int, default=1, help="MMseqs2 coverage mode (default: 1)")
    args = parser.parse_args()
    main(args.fasta, args.out_dir, args.train_size, args.val_size, args.test_size, args.mmseqs_threshold, args.cov, args.cov_mode)
