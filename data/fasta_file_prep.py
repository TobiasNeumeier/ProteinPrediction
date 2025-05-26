import gzip
from Bio import SeqIO
from collections import defaultdict

# STEP 1: Load Pfam-UniProt mappings
pfam_map = defaultdict(set)

base_path = "E:/Data/"
pfam_file_path = base_path + "Pfam-A.full.gz"
input_fasta = base_path + "uniprot_sprot.fasta.gz"

print("Reading Pfam mappings from Pfam-A.full.uniprot.gz...")
with gzip.open(pfam_file_path, "rt") as pfam_file:
    for line in pfam_file:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) >= 2:
            uniprot_id, pfam_id = parts[0], parts[1]
            pfam_map[uniprot_id].add(pfam_id)

print(f"Loaded Pfam domains for {len(pfam_map)} unique UniProt accessions.")

# STEP 2: Read Swiss-Prot FASTA and add Pfam annotations
output_fasta = base_path + "swissprot_with_pfam.fasta"

print("Processing Swiss-Prot FASTA and adding Pfam annotations...")
with gzip.open(input_fasta, "rt") as fasta_in, open(output_fasta, "w") as fasta_out:
    count_total = 0
    count_with_pfam = 0
    for record in SeqIO.parse(fasta_in, "fasta"):
        count_total += 1
        # Extract UniProt accession (e.g., sp|Q9Y261|...)
        parts = record.id.split("|")
        uniprot_id = parts[1] if len(parts) > 1 else record.id

        pfams = pfam_map.get(uniprot_id)
        if pfams:
            pfam_str = ",".join(sorted(pfams))
            record.description += f" Pfam={pfam_str}"
            count_with_pfam += 1
        else:
            record.description += " Pfam=None"

        SeqIO.write(record, fasta_out, "fasta")

print(f"\nDone! Output written to: {output_fasta}")
print(f"Total proteins processed: {count_total}")
print(f"Proteins with Pfam domains: {count_with_pfam}")
