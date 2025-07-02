import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import os, sys
sys.path.append(os.path.abspath('../models'))
from Dataset import ProteinCSVWindowDataset, ProteinEmbeddingWithLabelsDataset,  collate_fn_window
from original import OriginalModel, OriginalModelLarger
from small import SmallModel

CHECKPOINT_PATH = './checkpoints/original_bs8_lr3e-04_ep15_20250623_232208.pth'
TEST_DATA_PATH = r"E:\Data\embeddings41k\test\all_embeddings_and_labels.h5"
OUTPUT_CSV = 'predictions_emb_23.csv'
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Modell und Testdaten laden
model = OriginalModel(input_size=1024, channels=512, num_classes=58)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

test_dataset = ProteinEmbeddingWithLabelsDataset(TEST_DATA_PATH)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_window)

results = []

with torch.no_grad():
    for batch in test_loader:
        inputs = batch["embeddings"]
        targets = batch["labels"]
        inputs = inputs.to(DEVICE)
        outputs, _ = model(inputs)
        preds = outputs.argmax(dim=-1).cpu().numpy()
        targets = targets.cpu().numpy()
        for t, p in zip(targets, preds):
            results.append({'target': np.array2string(t, separator=","), 'prediction': np.array2string(p, separator=",")})

# In CSV schreiben
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f'Vorhersagen gespeichert in {OUTPUT_CSV}')