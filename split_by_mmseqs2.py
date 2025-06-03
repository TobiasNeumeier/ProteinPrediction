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
    out_dir = out_dir / fasta_path_base.replace('.fasta', '')
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

    # MMseqs2: use a dedicated subfolder for all intermediate files
    mmseqs_dir = out_dir / "mmseqs_work"
    mmseqs_dir.mkdir(exist_ok=True)
    db_train = mmseqs_dir / "train_db"
    db_temp = mmseqs_dir / "temp_db"
    aln_dir = mmseqs_dir / "tmp"
    aln_dir.mkdir(exist_ok=True)
    result_file = mmseqs_dir / "train_vs_temp.m8"

    subprocess.run(["mmseqs", "createdb", str(out_dir / "train_raw.fasta"), str(db_train)], check=True)
    subprocess.run(["mmseqs", "createdb", str(out_dir / "temp.fasta"), str(db_temp)], check=True)
    subprocess.run([
        "mmseqs", "search", str(db_train), str(db_temp), str(mmseqs_dir / "result"), str(aln_dir),
        "--min-seq-id", str(mmseqs_threshold)
    ], check=True)
    subprocess.run([
        "mmseqs", "convertalis", str(db_train), str(db_temp), str(mmseqs_dir / "result"), str(result_file),
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

    # Save to CSV (only these are in the main output folder)
    filtered_train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    # # Optionally, remove intermediate FASTA files
    # (out_dir / "train_raw.fasta").unlink(missing_ok=True)
    # (out_dir / "temp.fasta").unlink(missing_ok=True)

    # Print achieved ratios
    n_total = len(fasta_df)
    n_train = len(filtered_train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    print(f"Final split sizes: train={n_train} ({n_train/n_total:.2%}), val={n_val} ({n_val/n_total:.2%}), test={n_test} ({n_test/n_total:.2%})")