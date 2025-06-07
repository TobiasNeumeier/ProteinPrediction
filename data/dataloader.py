import torch
from torch.utils.data import Dataset
import pandas as pd

class ProteinResidueDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        # Use fam_id and fam_id_num columns
        self.num_fams = self.df['fam_id_num'].max()
        self.fam_id_to_num = {row['fam_id']: row['fam_id_num'] for _, row in self.df.iterrows()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        length = int(row['length'])
        fam_idx = int(row['fam_id_num']) - 1  # zero-based index
        label_mask = torch.zeros((length, self.num_fams), dtype=torch.long)
        start = int(row['start']) - 1  # 0-based
        end = int(row['end'])
        label_mask[start:end + 1, fam_idx] = 1  # one-hot for family in domain region
        return {
            'accession': row['AC'],
            'label_str': row['label'],
            'label': fam_idx,
            'residue_labels': label_mask,
            'length': length,
            "start": row['start'],
            "end": row['end'],
            'sequence': row['sequence'],
        }