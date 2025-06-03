import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from Bio import SeqIO

def parse_fasta(fasta_path):
    records = list(SeqIO.parse(fasta_path, "fasta"))
    return pd.DataFrame({
        "AC": [rec.id.split('|')[0] for rec in records],
        "label": [rec.id.split('|')[1].replace("label=", "", 1) for rec in records],  # TODO check for correctness
        "header": [rec.id for rec in records],
        "sequence": [str(rec.seq) for rec in records]
    })

def main(fasta_path, graphpart_csv, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assignments = pd.read_csv(graphpart_csv)
    fasta_df = parse_fasta(fasta_path)
    assignments = assignments.merge(fasta_df, on="AC", how="left")

    # Assign splits
    assignments['split'] = None
    assignments.loc[assignments['cluster'] == 0.0, 'split'] = 'train'

    if assignments['cluster'].nunique() == 2:
        cluster1 = assignments[assignments['cluster'] == 1.0].copy()
        if len(cluster1) > 1:
            val, test = train_test_split(cluster1, test_size=1/3, random_state=42)
            assignments.loc[val.index, 'split'] = 'val'
            assignments.loc[test.index, 'split'] = 'test'
        elif len(cluster1) == 1:
            assignments.loc[cluster1.index, 'split'] = 'val'
    else:
        assignments.loc[assignments['cluster'] == 1.0, 'split'] = 'val'
        assignments.loc[assignments['cluster'] == 2.0, 'split'] = 'test'

    # Save splits with sequence
    for split in ['train', 'val', 'test']:
        df_split = assignments[assignments['split'] == split]
        if not df_split.empty:
            df_split.to_csv(out_dir / f"{split}.csv", index=False)

    # Optionally, save val_and_test together
    # val_and_test = assignments[assignments['split'] != 'train']
    # if not val_and_test.empty:
    #     val_and_test.to_csv(out_dir / "val_and_test.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split FASTA/GraphPart assignments into train/val/test with sequence mapping.")
    parser.add_argument("--fasta", required=True, help="Path to input FASTA file")
    parser.add_argument("--graphpart_csv", required=True, help="Path to GraphPart clustered CSV")
    parser.add_argument("--out_dir", required=True, help="Output directory for split CSVs")
    args = parser.parse_args()
    main(args.fasta, args.graphpart_csv, args.out_dir)


# python split_by_graphpart.py --fasta data/fasta/pfam_subset_graphpart_400x40.fasta --graphpart_csv data/clustered/graphpart_assignments_partitions_3_th5.csv --out_dir ~/split/pfam_subset_graphpart_4_partitions_3_th_5
# graphpart mmseqs2 --fasta-file data/fasta/pfam_subset_graphpart_400x40.fasta --threshold 0.3 --out-file data/clustered/graphpart_assignments_400x40_partitions_3_th3.csv --labels-name label --partitions 3