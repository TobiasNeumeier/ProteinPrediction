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
    sequences = [item['sequence'] for item in batch]
    return {
        'accession': accessions,
        'label': labels,
        'residue_labels': residue_labels,
        'length': lengths,
        'sequence': sequences
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
            sequence = batch['sequence'][i]
            length = batch['length'][i].item()
            #raw_seq = "".join(map(str, sequence[:length].tolist()))
            raw_seq = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))  # re.sub(r"[UZOB]", "X", raw_seq)
            seq = "<AA2fold> " + " ".join(list(raw_seq))
            raw_seqs.append(seq)
            lengths.append(length)
            if i == 4:
                print(f"E.g. seq: {seq} with length {length} for accession {batch['accession'][i]}")
        
        # Tokenize with padding
        tokens = tokenizer.batch_encode_plus(
            raw_seqs, 
            return_tensors="pt", 
            padding="longest",  # Pad to longest sequence in batch
            add_special_tokens=True  # Adds EOS token!
        ).to(device)
        
        # Get the embeddings
        with torch.no_grad():
            output = model(**tokens).last_hidden_state

        for i, accession in enumerate(batch['accession']): 
            # Remove prefix token and padding
            emb = output[i, 1:lengths[i]+1]  
            labels = batch['residue_labels'][i][:lengths[i]]
            torch.save(emb.cpu(), proc_dir / f"{accession}_embedding.pt")
            torch.save(labels.cpu(), proc_dir / f"{accession}_labels.pt")
            # print only once as an example one protein, embedded and shape
            if i == 4:
                print(f"E.g.: Embedded protein of length {lengths[i]} to emb. of shape: {emb.shape}. labels: {labels.shape}. seq: {accession}")


def main(dataset, split, out_dir, batch_size=8):
    splits = [split] if split != "all" else ["train", "val", "test"]
    for split_name in splits:
        csv_path = Path(f"{dataset}/{split_name}.csv")
        proc_dir = Path(out_dir) / split_name
        proc_dir.mkdir(parents=True, exist_ok=True)
        sys.path.append(str(Path(__file__).parent / "data"))
        from dataloader import ProteinResidueDataset
        dataset_obj = ProteinResidueDataset(str(csv_path))
        sampler = SequentialSampler(dataset_obj)
        loader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=False, sampler=sampler, collate_fn=collate_fn)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer, model = load_prott5(device)
        generate_embeddings(loader, tokenizer, model, device, proc_dir)
        print(f"Embeddings for {split_name} set of {dataset} saved to {proc_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ProtT5 embeddings for a dataset split (train/val/test/all). E.g.: nice python create_prott5_embeddings.py --dataset data/split/mmseqs2/checkpoint_41646 --split all --out_dir data/embeddings/checkpoint_41646")
    parser.add_argument("--dataset", required=True,default="data/split/mmseqs2/checkpoint_41646",  help="Dataset path (e.g. data/split/mmseqs2/checkpoint_41646)")
    parser.add_argument("--split", required=True, choices=["train", "val", "test", "all"], help="Which split to process or 'all' for all splits")
    parser.add_argument("--out_dir", required=True, help="Output directory for embeddings")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for embedding generation")
    args = parser.parse_args()
    main(args.dataset, args.split, args.out_dir, args.batch_size)
