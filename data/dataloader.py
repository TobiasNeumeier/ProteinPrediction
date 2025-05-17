import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast


class ProteinResidueDataset(Dataset):
    def __init__(self, csv_path, label_to_index=None):
        self.df = pd.read_csv(csv_path)
        self.df['fragments'] = self.df['fragments'].apply(ast.literal_eval)
        self.df['family'] = 'PF01370'
        
        # Create label -> index mapping if not given
        if label_to_index is None:
            families = sorted(self.df['family'].unique())
            self.label_to_index = {fam: idx for idx, fam in enumerate(families)}
        else:
            self.label_to_index = label_to_index
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        length = int(row['length'])
        
        # Per-residue labels: 1 for family domain, 0 for background
        label_mask = torch.zeros(length, dtype=torch.long)
        for fragment in row['fragments']:
            start = fragment['start']
            end = fragment['end']
            label_mask[start:end + 1] = 1
        
        # Later on, we will save the embeddings (and one-hot encodings) and load them instead
        return {
            'accession': row['accession'],
            'label': self.label_to_index[row['family']],
            'residue_labels': label_mask,
            'length': length
        }
