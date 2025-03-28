import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
from conformer import ConformerEEG
from balanced_dataset import BalancedSeizureDataset, BalancedSeizureDataset_from_List
import glob
import random

# Focal Loss (Handles Class Imbalance)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        downsampled_targets = targets[:, ::4]  # Downsample by taking every 4th column
        bce_loss = self.bce(logits, downsampled_targets)
        pt = torch.exp(-bce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return loss.mean()


class cross_entropy(nn.Module):
    def __init__(self, downsample_rate=4):
        super(cross_entropy, self).__init__()
        self.downsample_rate = downsample_rate
        
    def forward(self, logits, targets):
        downsampled_targets = targets[:, ::self.downsample_rate]  # Downsample by taking every 4th column
        loss = nn.CrossEntropyLoss()(logits, downsampled_targets)
        return loss

def get_latest_model_from_dir(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not checkpoints:
        raise FileNotFoundError("❌ No model checkpoint found!")

    latest_checkpoint = max(checkpoints, key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

    return checkpoint_path



# Training Function
def train_model(model, train_loader, val_loader, model_dir=None, num_epochs=2000, lr=1e-3, device="cuda"):
    if model_dir is not None:
        checkpoint_path = get_latest_model_from_dir(model_dir)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Reduce LR every 10 epochs
    # criterion = FocalLoss(alpha=0.25, gamma=2)  # Use Focal Loss for imbalance
    criterion = cross_entropy()

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        for data, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            data, label = data.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(data)  # Output: (B, Seq)

            loss = criterion(output, label)  # Compute loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, label in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                data, label = data.to(device), label.to(device)
                output = model(data)
                loss = criterion(output, label)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # Print results
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")


        if epoch % 10 == 0:
        # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                sv_fn = os.path.join(model_dir, "epoch_%d_conformer_eeg.pth" % epoch)
                torch.save(model.state_dict(), sv_fn)
                print("🔥 Model checkpoint saved!")

        # Adjust learning rate
        scheduler.step()

def prepare_dataset(data_dir, split_ratio=(0.7, 0.2, 0.1)):
    all_eeg_npy_files = glob.glob(os.path.join(data_dir, "*.npy"))
    num_files = len(all_eeg_npy_files)
    num_train = int(split_ratio[0] * num_files)
    num_val = int(split_ratio[1] * num_files)
    
    # Shuffle the files randomly
    random.shuffle(all_eeg_npy_files)
    
    train_files = all_eeg_npy_files[:num_train]
    val_files = all_eeg_npy_files[num_train:num_train+num_val]
    test_files = all_eeg_npy_files[num_train+num_val:]
    return train_files, val_files, test_files

if __name__ == "__main__":
    data_dir = "/Users/jjiang10/Data/EEG/BIDS_Siena_balanced"
    save_dir = "/Users/jjiang10/Data/EEG/model_save_dir"

    train_files, val_files, test_files = prepare_dataset(data_dir)
    # Load Data & Train Model
    train_data = BalancedSeizureDataset_from_List(train_files)
    val_data = BalancedSeizureDataset_from_List(val_files)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=4)

    # Initialize Model
    model = ConformerEEG()

    # Train
    train_model(model, train_loader, val_loader, save_dir, num_epochs=2000, lr=1e-3, device="cuda" if torch.cuda.is_available() else "cpu")




