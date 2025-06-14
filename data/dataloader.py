import torch
from torch.utils.data import Dataset
import pandas as pd

class ProteinResidueDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        # Use fam_id and fam_id_num columns
        self.num_fams = self.df['fam_id_num'].nunique()
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
            'fam_id': row['fam_id'],
            'label': fam_idx,
            'residue_labels': label_mask,
            'length': length,
            "start": row['start'],
            "end": row['end'],
            'sequence': row['sequence'],
        }
    

class ProteinEmbeddingWithLabelsDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.file = None
        with h5py.File(self.h5_path, 'r') as f:
            self.keys = list(f['embeddings'].keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')

        accession = self.keys[idx]
        emb = self.file["embeddings"][accession][()]
        label = self.file["labels"][accession][()]

        emb_tensor = torch.tensor(emb, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "accession": accession,
            "embedding": emb_tensor,
            "label": label_tensor,
            "length": emb_tensor.shape[0]
        }

    def __del__(self):
        if self.file:
            self.file.close()

# with respective 
def collate_fn(batch):
    embeddings = [item["embedding"] for item in batch]
    labels = [item["label"] for item in batch]
    lengths = [item["length"] for item in batch]
    accessions = [item["accession"] for item in batch]

    padded_embeddings = pad_sequence(embeddings, batch_first=True)
    padded_labels = pad_sequence(labels, batch_first=True)

    return {
        "accessions": accessions,
        "embeddings": padded_embeddings,
        "labels": padded_labels,
        "lengths": torch.tensor(lengths)
        }