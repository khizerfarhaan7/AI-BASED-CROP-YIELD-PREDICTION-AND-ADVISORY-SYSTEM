import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from data.torch_dataset import CropYieldDataset
from models.lstm_yield import LSTMYieldModel

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. GPU is required for training.")

device = torch.device("cuda")

# Initialize dataset with partial week support (weeks 8-16)
# Dataset will randomly select start week between 8-16 during training
full_dataset = CropYieldDataset("data/X_final.npy", "data/y_final.npy", 
                                min_start_week=8, max_start_week=16, final_week=16)

# Split dataset into train (80%) and test (20%)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    
    Logic:
    - Sequences have different lengths (depending on random start week)
    - Pad sequences to same length for batching
    - Return lengths for pack_padded_sequence in model
    """
    X_seqs, y_labels, lengths = zip(*batch)
    
    # Pad sequences to same length (longest in batch)
    X_padded = pad_sequence(X_seqs, batch_first=True)
    y_batch = torch.stack(y_labels)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    
    return X_padded, y_batch, lengths_tensor

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

model = LSTMYieldModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    
    for X_batch, y_batch, lengths in train_dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        # Keep lengths on CPU for pack_padded_sequence
        lengths = lengths.cpu()
        
        optimizer.zero_grad()
        # Pass sequence lengths to model for masking future timesteps
        outputs = model(X_batch, lengths)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Save the trained model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/model.pth")
print("✅ Model saved successfully to models/model.pth")

# Evaluate model on test set
print("\n" + "="*50)
print("Model Evaluation")
print("="*50)

model.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    for X_batch, y_batch, lengths in test_dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        lengths = lengths.cpu()
        
        outputs = model(X_batch, lengths)
        
        all_predictions.append(outputs.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())

# Concatenate all predictions and targets
all_predictions = np.concatenate(all_predictions)
all_targets = np.concatenate(all_targets)

# Compute evaluation metrics
mae = np.mean(np.abs(all_predictions - all_targets))
rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))

# Display evaluation results
print(f"\nTest Set Evaluation Metrics:")
print(f"  Mean Absolute Error (MAE): {mae:.4f}")
print(f"  Root Mean Square Error (RMSE): {rmse:.4f}")
print(f"\nTest samples evaluated: {len(all_targets)}")
print("="*50)
