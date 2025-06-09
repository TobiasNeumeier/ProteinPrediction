import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_seqence
import h5py


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