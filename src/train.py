import os
import sys
import datetime
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import optuna
import wandb
from tqdm import tqdm

# Model and Dataset imports
sys.path.append(os.path.abspath('../models'))
from Dataset import ProteinCSVWindowDataset, collate_fn_window
from original import OriginalModel
from small import SmallModel


# ----------------------------
# Configuration & Registry
# ----------------------------
MODEL_REGISTRY = {
    "original": OriginalModel,
    "small": SmallModel,
}

config = {
    "model_name": "original",
    "num_classes": 2,
    "batch_size": 1,
    "epochs": 100,
    "learning_rate": 1e-3,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "sequence_max_len": 512,
}


# ----------------------------
# Utility Functions
# ----------------------------
def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_model(model, config, directory="checkpoints"):
    os.makedirs(directory, exist_ok=True)
    base_name = (
        f"{config['model_name']}_bs{config['batch_size']}_"
        f"lr{config['learning_rate']:.0e}_ep{config['epochs']}"
    )
    # Delete existing models with same config
    for fname in os.listdir(directory):
        if fname.startswith(base_name) and fname.endswith(".pth"):
            os.remove(os.path.join(directory, fname))
            print(f"🗑️ Deleted old checkpoint: {fname}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(directory, f"{base_name}_{timestamp}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to {save_path}")
    return save_path

def compute_token_accuracy(outputs, targets, threshold=0.7):
    target_indices = targets.argmax(dim=-1)
    probs = F.softmax(outputs, dim=-1)
    max_probs, pred_indices = probs.max(dim=-1)
    pred_indices[max_probs < threshold] = -1
    correct = (pred_indices == target_indices) & (pred_indices != -1)
    total = (pred_indices != -1).sum().item()
    return 0.0 if total == 0 else correct.sum().item() / total

def evaluate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Evaluating", leave=False)
        for batch in progress_bar:
            inputs = batch["embeddings"].to(config["device"])
            targets = batch["labels"].to(config["device"])
            outputs, _ = model(inputs)
            val_acc += compute_token_accuracy(outputs, targets)
            logits = outputs.view(-1, outputs.size(-1))
            targets = targets.argmax(dim=-1).view(-1)
            loss = criterion(logits, targets)
            val_loss += loss.item()
    return val_loss / len(val_loader), val_acc / len(val_loader)


def train_model(model, dataloader, val_loader, config, run, criterion, optimizer, max_epochs, patience):
    best_val_loss = float('inf')
    patience_counter = 0
    quarter_epoch = max(1, int(len(dataloader) / 4))

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{max_epochs}", leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            inputs = batch["embeddings"].to(config["device"])
            targets = batch["labels"].to(config["device"])

            optimizer.zero_grad()
            outputs, _ = model(inputs)
            running_acc += compute_token_accuracy(outputs, targets)
            logits = outputs.view(-1, outputs.size(-1))
            targets = targets.argmax(dim=-1).view(-1)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            if (batch_idx + 1) % quarter_epoch == 0:
                val_loss, val_acc = evaluate(model, val_loader, criterion)
                print(f"[Batch {batch_idx}] Mid-Epoch Validation Loss: {val_loss:.4f} | ValAcc: {val_acc:.4f}")
                run.log({"val_loss": val_loss, "val_acc": val_acc})
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    save_model(model, config)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping triggered (within epoch).")
                        return best_val_loss

        avg_loss = running_loss / len(dataloader)
        avg_acc = running_acc / len(dataloader)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}: TrainLoss={avg_loss:.4f}, ValLoss={val_loss:.4f}, ValAcc={val_acc:.4f}")
        run.log({"train_loss": avg_loss, "train_acc": avg_acc, "val_loss": val_loss, "val_acc": val_acc})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_model(model, config)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered (epoch end).")
                break
    return best_val_loss


# ----------------------------
# Execution Entry Point
# ----------------------------
if __name__ == "__main__":
    seed_everything()
    dataset = ProteinCSVWindowDataset("./train.csv")
    valDataset = ProteinCSVWindowDataset("./val.csv")

    def objective(trial):
        config.update({
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [1, 2, 4, 8]),
            "model_name": trial.suggest_categorical("model_name", ["original"]),
            "epochs": 15,
        })
        run = wandb.init(entity="protpred", project="protpred", config=config)
        dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn_window)
        val_loader = DataLoader(valDataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn_window)
        model = MODEL_REGISTRY[config["model_name"]](num_classes=dataset.num_labels).to(config["device"])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        best_loss = train_model(model, dataloader, val_loader, config, run, criterion, optimizer, config["epochs"], patience=5)
        run.finish()
        return best_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)
    best_params = study.best_trial.params
    config.update({
        "learning_rate": best_params["learning_rate"],
        "batch_size": best_params["batch_size"],
        "model_name": best_params["model_name"],
        "epochs": 1000,
    })

    final_run = wandb.init(entity="protpred", project="protpred", config=config)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn_window)
    val_loader = DataLoader(valDataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn_window)
    model = MODEL_REGISTRY[config["model_name"]](num_classes=dataset.num_labels).to(config["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    train_model(model, dataloader, val_loader, config, final_run, criterion, optimizer, config["epochs"], patience=30)
    final_run.finish()
