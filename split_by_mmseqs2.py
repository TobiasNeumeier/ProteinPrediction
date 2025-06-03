import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from Bio import SeqIO
import subprocess

def parse_fasta(fasta_path):
    records = list(SeqIO.parse(fasta_path, "fasta"))
    return pd.DataFrame({
        "AC": [rec.id.split('|')[0] for rec in records],
        "label": [rec.id.split('|')[1].replace("label=", "", 1) for rec in records],
        "header": [rec.id for rec in records],
        "sequence": [str(rec.seq) for rec in records]
    })

def write_fasta(df, path):
    with open(path, "w") as f:
        for _, row in df.iterrows():
            f.write(f">{row['header']}\n{row['sequence']}\n")

def main(fasta_path, out_dir, train_size, val_size, test_size, mmseqs_threshold):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fasta_path_base = Path(fasta_path).name
    out_dir = Path(out_dir + fasta_path_base.replace('.fasta', ''))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    fasta_df = parse_fasta(fasta_path)

    # Normalize ratios
    total = train_size + val_size + test_size
    train_ratio = train_size / total
    val_ratio = val_size / total
    test_ratio = test_size / total

    # First split: train vs. temp (val+test)
    train_df, temp_df = train_test_split(fasta_df, test_size=(1-train_ratio), random_state=42)

    # Write train and temp to FASTA for MMseqs2
    write_fasta(train_df, out_dir / "train_raw.fasta")
    write_fasta(temp_df, out_dir / "temp.fasta")

    # MMseqs2: find train sequences similar to temp
    db_train = out_dir / "train_db"
    db_temp = out_dir / "temp_db"
    aln_dir = out_dir / "mmseqs_tmp"
    aln_dir.mkdir(exist_ok=True)
    result_file = out_dir / "train_vs_temp.m8"

    subprocess.run(["mmseqs", "createdb", str(out_dir / "train_raw.fasta"), str(db_train)], check=True)
    subprocess.run(["mmseqs", "createdb", str(out_dir / "temp.fasta"), str(db_temp)], check=True)
    subprocess.run([
        "mmseqs", "search", str(db_train), str(db_temp), str(out_dir / "result"), str(aln_dir),
        "--min-seq-id", str(mmseqs_threshold)
    ], check=True)
    subprocess.run([
        "mmseqs", "convertalis", str(db_train), str(db_temp), str(out_dir / "result"), str(result_file),
        "--format-output", "query,target,pident"
    ], check=True)

    # Remove from train any sequence that matches temp above threshold
    similar_train_ACs = set()
    with open(result_file) as f:
        for line in f:
            query, target, pident = line.strip().split('\t')
            if float(pident) >= mmseqs_threshold * 100:
                similar_train_ACs.add(query.split('|')[0])
    filtered_train_df = train_df[~train_df['AC'].isin(similar_train_ACs)]

    # Now split temp into val and test
    val_relative = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(temp_df, test_size=(1-val_relative), random_state=42)

    # Save to CSV
    filtered_train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    # Print achieved ratios
    n_total = len(fasta_df)
    n_train = len(filtered_train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    print(f"Final split sizes: train={n_train} ({n_train/n_total:.2%}), val={n_val} ({n_val/n_total:.2%}), test={n_test} ({n_test/n_total:.2%})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split FASTA with MMseqs2 deduplication between train and temp (val+test).E.g.: python split_by_mmseqs2.py \
  --fasta data/fasta/pfam_subset_graphpart_400x40.fasta \
  --out_dir data/split/mmseqs2 \
  --train_size 0.7 \
  --val_size 0.2 \
  --test_size 0.1 \
  --mmseqs_threshold 0.3")
    parser.add_argument("--fasta", required=True, help="Path to input FASTA file")
    parser.add_argument("--out_dir", required=True, help="Output directory for split CSVs")
    parser.add_argument("--train_size", type=float, default=0.7, help="Proportion for train set (default: 0.7)")
    parser.add_argument("--val_size", type=float, default=0.2, help="Proportion for val set (default: 0.2)")
    parser.add_argument("--test_size", type=float, default=0.1, help="Proportion for test set (default: 0.1)")
    parser.add_argument("--mmseqs_threshold", type=float, default=0.3, help="MMseqs2 min-seq-id threshold (default: 0.3)")
    args = parser.parse_args()
    main(args.fasta, args.out_dir, args.train_size, args.val_size, args.test_size, args.mmseqs_threshold)