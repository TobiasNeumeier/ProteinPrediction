import sys
import os
sys.path.append(os.path.abspath('../models'))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import optuna


# ----------------------------------
# 🧠 MODEL REGISTRY
# ----------------------------------
from original import OriginalModel
from small import SmallModel

MODEL_REGISTRY = {
    "original": OriginalModel,
    "small": SmallModel,
}


# ----------------------------------
# ⚙️ CONFIGURATION
# ----------------------------------
config = {
    "model_name": "original",
    "num_classes": 2,
    "batch_size": 1,
    "epochs": 100,
    "learning_rate": 1e-3,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "sequence_max_len": 512,
}


# ----------------------------------
# 🔢 ENCODING UTILS
# ----------------------------------

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY-"  # 20 + gap
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def one_hot_encode_sequence(seq, max_len=512):
    tensor = torch.zeros((max_len, len(AMINO_ACIDS)))
    for i, aa in enumerate(seq[:max_len]):
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


# ----------------------------------
# 📦 DATASET + DATALOADER
# ----------------------------------
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, max_len):
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = one_hot_encode_sequence(self.sequences[idx], self.max_len)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return seq, label


# Dummy data
sequences = ["MSSDTHGTDLADGDVLVTGAAGFIGSHLVTELRNSGRNVVAVDRRPLPDDLESTSPPFTGSLREIRGDLNSLNLVDCLKNISTVFHLAALPGVRPSWTQFPEYLRCNVLATQRLMEACVQAGVERVVVASSSSVYGGADGVMSEDDLPRPLSPYGVTKLAAERLALAFAARGDAELSVGALRFFTVYGPGQRPDMFISRLIRATLRGEPVEIYGDGTQLRDFTHVSDVVRALMLTASVRDRGSAVLNIGTGSAVSVNEVVSMTAELTGLRPCTAYGSARIGDVRSTTADVRQAQSVLGFTARTGLREGLATQIEWTRRSLSGAEQDTVPVGGSSVSVPRL"]
labels = [[1 if 15 <= i <= 249 else 0 for i in range(len(sequences[0]))]]

dataset = SequenceDataset(sequences, labels, len(sequences[0]))


# ----------------------------------
# 🚂 TRAINING LOOP
# ----------------------------------
def train(model, dataloader, config):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    model.to(config["device"])

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(config["device"])
            targets = targets.to(config["device"])

            optimizer.zero_grad()
            logits, _ = model(inputs)

            logits = logits.view(-1, config["num_classes"])
            targets = targets.view(-1)

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{config['epochs']} - Loss: {avg_loss:.4f}")

def objective(trial):
    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [1, 2, 4])
    model_name = trial.suggest_categorical("model_name", ["original"])
    max_epochs = 30  # Shorter for tuning

    # Update config
    config.update({
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "model_name": model_name,
        "epochs": max_epochs,
    })

    # Recreate dataloader with new batch size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate model
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(num_classes=config["num_classes"]).to(config["device"])

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in dataloader:
            inputs = inputs.to(config["device"])
            targets = targets.to(config["device"])

            optimizer.zero_grad()
            outputs, _ = model(inputs)

            outputs = outputs.view(-1, config["num_classes"])
            targets = targets.view(-1)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)

        # Report to Optuna (used for pruning)
        trial.report(avg_loss, epoch)

        # Early stopping with Optuna pruning
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Manual early stopping (optional)
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break  # Stop training early

    return best_val_loss



# ----------------------------------
# 🧪 EXECUTION
# ----------------------------------
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("Best trial:")
    print("  Value: ", study.best_trial.value)
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")


    # Extract best hyperparameters
    best_params = study.best_trial.params
    config.update(best_params)
    config["epochs"] = 100  # train longer now

    # Rebuild dataloader (in case batch size changed)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Retrain final model
    final_model_class = MODEL_REGISTRY[config["model_name"]]
    final_model = final_model_class(num_classes=config["num_classes"])
    train(final_model, dataloader, config)
