import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from Bio import SeqIO

def parse_fasta(fasta_path):
    records = list(SeqIO.parse(fasta_path, "fasta"))
    return pd.DataFrame({
        "AC": [rec.id.split('|')[0] for rec in records],
        "label": [rec.id.split('|')[1].replace("label=", "", 1) for rec in records],
        "header": [rec.id for rec in records],
        "sequence": [str(rec.seq) for rec in records]
    })

def main(fasta_path, graphpart_csv, out_dir, train_size, val_size, test_size):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assignments = pd.read_csv(graphpart_csv)
    fasta_df = parse_fasta(fasta_path)
    assignments = assignments.merge(fasta_df, on="AC", how="left")

    # Normalize ratios
    total = train_size + val_size + test_size
    train_ratio = train_size / total
    val_ratio = val_size / total
    test_ratio = test_size / total

    # First split: train vs. temp (val+test)
    train_df, temp_df = train_test_split(assignments, test_size=(1-train_ratio), random_state=42)
    # Second split: temp into val and test
    val_relative = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(temp_df, test_size=(1-val_relative), random_state=42)

    # Save to CSV
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    # Print achieved ratios
    n_total = len(assignments)
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    print(f"Final split sizes: train={n_train} ({n_train/n_total:.2%}), val={n_val} ({n_val/n_total:.2%}), test={n_test} ({n_test/n_total:.2%})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flexible train/val/test split for FASTA/GraphPart assignments with sequence mapping. E.g.: python split_by_graphpart.py --fasta data/fasta/pfam_subset_graphpart_400x40.fasta --graphpart_csv data/clustered/graphpart_assignments_partitions_3_th5.csv --out_dir ./split --train_size 0.7 --val_size 0.15 --test_size 0.15")
    parser.add_argument("--fasta", required=True, help="Path to input FASTA file")
    parser.add_argument("--graphpart_csv", required=True, help="Path to GraphPart clustered CSV")
    parser.add_argument("--out_dir", required=True, help="Output directory for split CSVs")
    parser.add_argument("--train_size", type=float, default=0.7, help="Proportion for train set (default: 0.7)")
    parser.add_argument("--val_size", type=float, default=0.15, help="Proportion for val set (default: 0.15)")
    parser.add_argument("--test_size", type=float, default=0.15, help="Proportion for test set (default: 0.15)")
    args = parser.parse_args()
    main(args.fasta, args.graphpart_csv, args.out_dir, args.train_size, args.val_size, args.test_size)