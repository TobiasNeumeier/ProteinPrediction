import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import h5py
import pandas as pd
import json


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



AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY-"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def one_hot_encode_sequence(seq):
    tensor = torch.zeros((len(seq), len(AMINO_ACIDS)))
    for i, aa in enumerate(seq):
        if aa in AA_TO_IDX:
            tensor[i, AA_TO_IDX[aa]] = 1.0
    return tensor

def one_hot_encode_labels(label_tensor, num_classes=2):
    if isinstance(label_tensor, list):
        label_tensor = torch.tensor(label_tensor)
    label_tensor = label_tensor.long()
    batch_size, seq_len = label_tensor.shape
    one_hot = torch.zeros((batch_size, seq_len, num_classes), dtype=torch.float)
    for b in range(batch_size):
        for i in range(seq_len):
            class_idx = label_tensor[b, i]
            if 0 <= class_idx < num_classes:
                one_hot[b, i, class_idx] = 1.0
    return one_hot

class ProteinCSVWindowDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        with open("./label_index.json", 'r') as f:
            self.label_to_index = json.load(f)

        self.num_labels = len(self.label_to_index)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        accession = row["AC"]
        sequence = row["sequence"]
        start = int(row["start"])
        end = int(row["end"])

        # Truncate sequence and label to max_len
        seq = sequence
        seq_tensor = one_hot_encode_sequence(seq)

        pf_label = row["label"].split(".")[0]
        label_idx = self.label_to_index[pf_label]

        # Generate binary label: 1 in [start:end], else 0
        label_tensor = torch.zeros((len(seq), self.num_labels), dtype=torch.float)
        for i in range(start-1, end):
            if i < len(sequence):
                label_tensor[i, label_idx] = 1.0


        return {
            "accession": accession,
            "embedding": seq_tensor,
            "label": label_tensor,
            "length": len(sequence)
        }
    
    




def collate_fn_window(batch):
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
