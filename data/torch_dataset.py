import numpy as np
import torch
from torch.utils.data import Dataset
import random

class CropYieldDataset(Dataset):
    """
    Dataset supporting variable-length sequences for partial week prediction.
    
    Logic:
    - Full sequences have 20 timesteps (weeks 1-20)
    - During training, randomly selects start week between 8-16
    - Returns sequence from start_week to week 16 (masking future timesteps)
    - Model still predicts final yield using partial sequence
    """
    def __init__(self, X_path, y_path, min_start_week=8, max_start_week=16, final_week=16):
        X = np.load(X_path)
        y = np.load(y_path)
        
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.min_start_week = min_start_week  # Week 8 (0-indexed: 7)
        self.max_start_week = max_start_week  # Week 16 (0-indexed: 15)
        self.final_week = final_week  # Last week to include (0-indexed: 15)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Randomly select start week between min_start_week and max_start_week
        # This creates variable-length sequences during training
        start_week = random.randint(self.min_start_week - 1, self.max_start_week - 1)  # Convert to 0-indexed
        
        # Extract sequence from start_week to final_week (inclusive)
        # This masks future timesteps after final_week
        X_seq = self.X[idx, start_week:self.final_week]
        
        # Return sequence length for packing (used in model)
        seq_length = X_seq.shape[0]
        
        return X_seq, self.y[idx], seq_length

if __name__ == "__main__":
    dataset = CropYieldDataset("data/X_final.npy", "data/y_final.npy")
    print(f"Dataset length: {len(dataset)}")
    
    X_sample, y_sample, seq_length = dataset[0]
    print(f"Sample X shape: {X_sample.shape}")
    print(f"Sample y shape: {y_sample.shape}")
    print(f"Sequence length: {seq_length}")

