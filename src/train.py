import datetime
import sys
import os

from tqdm import tqdm
sys.path.append(os.path.abspath('../models'))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import optuna
from Dataset import ProteinCSVWindowDataset, collate_fn_window
import torch.nn.functional as F


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
# 📦 DATASET
# ----------------------------------
dataset = ProteinCSVWindowDataset("./train.csv")
valDataset = ProteinCSVWindowDataset("./val.csv")


def save_model(model, config, directory="checkpoints"):
    os.makedirs(directory, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = (
        f"{config['model_name']}_"
        f"bs{config['batch_size']}_"
        f"lr{config['learning_rate']:.0e}_"
        f"ep{config['epochs']}_"
        f"{timestamp}.pth"
    )
    save_path = os.path.join(directory, name)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to {save_path}")
    return save_path


def compute_token_accuracy(outputs, targets, threshold = 0.7):
    """
    outputs: Tensor of shape [B, L, C] — logits from the model
    targets: Tensor of shape [B, L, C] — one-hot encoded true labels

    Returns: accuracy (float)
    """
    # Convert one-hot targets to class indices
    target_indices = targets.argmax(dim=-1)     # [B, L]
    
    # Convert logits to probabilities
    probs = F.softmax(outputs, dim=-1)          # [B, L, C]
    max_probs, pred_indices = probs.max(dim=-1) # [B, L], [B, L]

    # Reject low-confidence predictions
    pred_indices[max_probs < threshold] = -1

    # Compare only valid predictions
    correct = (pred_indices == target_indices) & (pred_indices != -1)
    total = (pred_indices != -1).sum().item()

    if total == 0:
        return 0.0

    accuracy = correct.sum().item() / total
    return accuracy

def evaluate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["embeddings"].to(config["device"])
            targets = batch["labels"].to(config["device"])
            outputs, _ = model(inputs)
            val_acc += compute_token_accuracy(outputs, targets)
            targets = targets.argmax(dim=-1)
            logits = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            loss = criterion(logits, targets)
            val_loss += loss.item()
    return val_loss / len(val_loader), val_acc / len(val_loader)








# ----------------------------------
# 🚂 TRAINING LOOP
# ----------------------------------


def objective(trial):
    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [1, 2, 4, 8])
    model_name = trial.suggest_categorical("model_name", ["original"])
    max_epochs = 15  # Shorter for tuning

    # Update config
    config.update({
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "model_name": model_name,
        "epochs": max_epochs,
    })

    # Recreate dataloader with new batch size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_window)

    # Instantiate model
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(num_classes=dataset.num_labels).to(config["device"])

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{max_epochs}", leave=False)
        for batch in progress_bar:
            #print(inputs)
            inputs = batch["embeddings"].to(config["device"])         # [B, L, 21]
            targets = batch["labels"].to(config["device"]) 

            #torch.set_printoptions(threshold=10000000)
            #print(inputs)
            #print(targets)
            #exit()

            optimizer.zero_grad()
            outputs, _ = model(inputs)

            running_acc += compute_token_accuracy(outputs, targets)

            #print(outputs.shape)
            #print(targets.shape)
            targets = targets.argmax(dim=-1)

            logits = outputs.view(-1, outputs.size(-1))  # → [B*L, 374]
            targets = targets.view(-1)    

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()


            

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_acc = running_acc / len(dataloader)
        avg_loss = running_loss / len(dataloader)
        print("Training")
        print(f"Epoch {epoch+1}/{config['epochs']} - Loss: {avg_loss:.4f} - AvgAcc: {avg_acc:.4f}")

        val_loader = DataLoader(valDataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_window)
        avg_loss, avg_acc = evaluate(model, val_loader, criterion)
        print(f"Validation Loss: {avg_loss:.4f} - ValAcc: {avg_acc:.4f}")


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
    study.optimize(objective, n_trials=2)

    print("Best trial:")
    print("  Value: ", study.best_trial.value)
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    best_params = study.best_trial.params

    config.update({
    "learning_rate": best_params["learning_rate"],
    "batch_size": best_params["batch_size"],
    "model_name": best_params["model_name"],
    "epochs": 1000,  # Or however long you want the final training to run
    })



    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn_window)

    # Instantiate model
    model_class = MODEL_REGISTRY[config["model_name"]]
    model = model_class(num_classes=dataset.num_labels).to(config["device"])

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0

    for epoch in range(100000):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{100000}", leave=False)
        for batch in progress_bar:
            #print(inputs)
            inputs = batch["embeddings"].to(config["device"])         # [B, L, 21]
            targets = batch["labels"].to(config["device"]) 

            #torch.set_printoptions(threshold=10000000)
            #print(inputs)
            #print(targets)
            #exit()

            optimizer.zero_grad()
            outputs, _ = model(inputs)

            running_acc += compute_token_accuracy(outputs, targets)

            #print(outputs.shape)
            #print(targets.shape)
            targets = targets.argmax(dim=-1)

            logits = outputs.view(-1, outputs.size(-1))  # → [B*L, 374]
            targets = targets.view(-1)    

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()


            

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_acc = running_acc / len(dataloader)
        avg_loss = running_loss / len(dataloader)
        print("Training")
        print(f"Epoch {epoch+1}/{config['epochs']} - Loss: {avg_loss:.4f} - AvgAcc: {avg_acc:.4f}")

        val_loader = DataLoader(valDataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn_window)
        avg_loss, avg_acc = evaluate(model, val_loader, criterion)
        print(f"Validation Loss: {avg_loss:.4f} - ValAcc: {avg_acc:.4f}")


        # Manual early stopping (optional)
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break  # Stop training early
    save_model(model, config)


