import re
import gzip
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from pathlib import Path
from api_connector import ApiConnector


def write_fasta(result_dict, output_path, output_name):
    """
    Writes the result_dict to a FASTA file in directory output_path with name output_name.
    Appends "_X" to the output_name where X is number of sequences.
    """
    output_name = output_name + f"_{len(result_dict)}.fasta"
    full_output_path = Path(output_path) / output_name
    with open(full_output_path, 'w') as out_f:
        for i, (acc, (pfam_id, start, end, full_seq)) in enumerate(result_dict.items()):
            if full_seq is None:
                continue  # skip if sequence wasn't found
            header = f">{acc}|label={pfam_id}-{start}-{end}|priority={i}"
            out_f.write(f"{header}\n{full_seq}\n")
    print(f"✅ Wrote {len(result_dict)} entries to {full_output_path}")
    

def run(api_connector: ApiConnector, directory, pfam_file, max_families, max_sequences_per_fam, checkpoint=2000):
    """
    Builds result_dict and writes it to a FASTA file.
    :param api_connector: the ApiConnector instance to fetch sequences
    :param directory: directory where output will be saved
    :param pfam_file: path to the pfam file
    :param max_families: maximum number of Pfam families to process
    :param max_sequences_per_fam: maximum number of sequences per Pfam family to process
    :param checkpoint: number of sequences to process before writing checkpoint to file
    :return:
    """
    result_dict = {}
    seq_count = 0
    pfam_file = Path(pfam_file)
    
    # One block is one Pfam family with multiple sequences
    for i, block in enumerate(pfam_block_iterator(pfam_file)):
        print(f"Processing block {i + 1}...")
        result = parse_pfam_block(block, max_seqs_per_fam=max_sequences_per_fam)
        for entry in result:
            pfam_id, acc, start, end = entry
            sequence = api_connector.fetch_sequence(acc)
            if sequence:
                result_dict[acc] = (pfam_id, start, end, sequence)
                seq_count += 1
            
            if seq_count % checkpoint == 0:
                write_fasta(result_dict, directory, "checkpoint")
    
        if i >= max_families:
            break


def pfam_block_iterator(filepath: Path):
    """
    Generator that yields blocks from a Pfam-A.full.gz file.
    Each block ends with a line containing only '//'.
    """
    with gzip.open(filepath, 'rt') as f:
        block = []
        for line in f:
            block.append(line)
            if line.strip() == '//':
                yield ''.join(block)  # return as single string
                block = []


def parse_pfam_block(block, max_seqs_per_fam=10):
    """
    Extract:
    - Pfam domain ID from '#=GF ID'
    - First X #=GS line's accession and start-end
    Returns (pfam_id, accession, start, end) or None if not found
    """
    pfam_id = None
    
    # Extract Pfam domain ID
    for line in block.splitlines():
        if line.startswith("#=GF AC"):
            pfam_id = line.strip().split()[2]  # line: "#=GF AC PF00069"
            break
    
    res = []
    # Extract first x matching #=GS line
    counter = 0
    for line in block.splitlines():
        if line.startswith("#=GS") and "AC" in line:
            match = re.match(r"#=GS\s+(\S+)/(\d+)-(\d+)\s+AC\s+(\S+)", line)
            if match:
                full_acc = match.group(1)  # e.g., A0A7D5S632_UNCBA
                start = int(match.group(2))  # 1
                end = int(match.group(3))  # 78
                acc = full_acc.split('_')[0]  # Strip organism suffix
                res.append((pfam_id, acc, start, end))
                
                counter += 1
                if counter == max_seqs_per_fam:
                    break
    return res


if __name__ == "__main__":
    api = ApiConnector(max_requests_per_minute=200 * 60)
    run(api, directory="E:/Data/", pfam_file="E:/Data/Pfam-A.full.gz", max_families=200, max_sequences_per_fam=1000, checkpoint=2000)
