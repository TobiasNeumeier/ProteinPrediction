import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm
import re
import os
import sys
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    accessions = [item['accession'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)
    residue_labels = pad_sequence([item['residue_labels'] for item in batch], batch_first=True, padding_value=0)
    return {
        'accession': accessions,
        'label': labels,
        'residue_labels': residue_labels,
        'length': lengths
    }

def load_prott5(device):
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False, legacy=True)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device)
    if device.type == "cuda":
        print("Moving model to GPU")
        model = model.half()
    else:
        print("Moving model to CPU - not using half precision")
    model.eval()
    return tokenizer, model

def generate_embeddings(loader, tokenizer, model, device, proc_dir):
    for batch in tqdm(loader, desc="Generating embeddings"):
        raw_seqs = []
        lengths = []
        # Prepare sequences for tokenization
        for i in range(len(batch['accession'])):
            sequence = batch['residue_labels'][i]
            length = batch['length'][i].item()
            raw_seq = "".join(map(str, sequence[:length].tolist()))
            raw_seq = re.sub(r"[UZOB]", "X", raw_seq)
            seq = "<AA2fold> " + " ".join(list(raw_seq))
            raw_seqs.append(seq)
            lengths.append(length)
        
        # Tokenize with padding
        tokens = tokenizer.batch_encode_plus(
            raw_seqs, 
            return_tensors="pt", 
            padding="longest",  # Pad to longest sequence in batch
            add_special_tokens=True  # Adds EOS token!
        ).to(device)
        
        with torch.no_grad():
            embedding_repr  = model(**tokens)
        
        # # Remove prefix & EOS from embeddings and mask
        # residue_embeddings = outputs.last_hidden_state[:, 1:-1]  # Remove <AA2fold> and </s>
        # mask = tokens.attention_mask[:, 1:-1]  # Corresponding mask
        
        for i, accession in enumerate(batch['accession']):
            # Extract non-padding embeddings
            emb = embedding_repr.last_hidden_state[i,1:lengths[i]+1]
            labels = batch['residue_labels'][i][:length]
            
            torch.save(emb.cpu(), proc_dir / f"{accession}_embedding.pt")
            torch.save(labels.cpu(), proc_dir / f"{accession}_labels.pt")

        tokens = tokenizer.batch_encode_plus(
            raw_seqs, return_tensors="pt", padding=True, add_special_tokens=True
        ).to(device)
        with torch.no_grad():
            output = model(**tokens).last_hidden_state
        for i, accession in enumerate(batch['accession']): 
            # Remove prefix token and padding
            emb = output[i, 1:lengths[i]+1]  
            labels = batch['residue_labels'][i][:lengths[i]]
            torch.save(emb.cpu(), proc_dir / f"{accession}_embedding.pt")
            torch.save(labels.cpu(), proc_dir / f"{accession}_labels.pt")
            # print only once as an example one protein, embedded and shape
            if i == 0:
                print(f"E.g.: Embedded protein of length {lengths[i]} to emb. of shape: {emb.shape}. labels: {labels.shape}. seq: {accession}")


def main(dataset, split, out_dir, batch_size=8):
    csv_path = Path(f"data/split/mmseqs2/{dataset}/{split}.csv")
    proc_dir = Path(out_dir) / dataset / split
    proc_dir.mkdir(parents=True, exist_ok=True)
    sys.path.append(str(Path(__file__).parent / "data"))
    from dataloader import ProteinResidueDataset
    dataset_obj = ProteinResidueDataset(str(csv_path))
    sampler = SequentialSampler(dataset_obj)
    loader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=False, sampler=sampler, collate_fn=collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_prott5(device)
    generate_embeddings(loader, tokenizer, model, device, proc_dir)
    print(f"Embeddings for {split} set of {dataset} saved to {proc_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ProtT5 embeddings for a dataset split (train/val/test)")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. pfam_subset_graphpart_400x40)")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"], help="Which split to process")
    parser.add_argument("--out_dir", required=True, help="Output directory for embeddings")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for embedding generation")
    args = parser.parse_args()
    main(args.dataset, args.split, args.out_dir, args.batch_size)
