import os
import csv
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from data.torch_dataset import CropYieldDataset
from models.lstm_yield import LSTMYieldModel

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model path
MODEL_PATH = "models/model.pth"

# Check if model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# Load dataset
print("\nLoading dataset...")
full_dataset = CropYieldDataset("data/X_final.npy", "data/y_final.npy", 
                                min_start_week=8, max_start_week=16, final_week=16)

# Split dataset into train (80%) and test (20%) with fixed seed for reproducibility
torch.manual_seed(42)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
print(f"Dataset size: {len(full_dataset)}")
print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

# Collate function for batching
def collate_fn(batch):
    X_seqs, y_labels, lengths = zip(*batch)
    X_padded = pad_sequence(X_seqs, batch_first=True)
    y_batch = torch.stack(y_labels)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    return X_padded, y_batch, lengths_tensor

# Create test dataloader
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

# Load model
print(f"\nLoading model from {MODEL_PATH}...")
model = LSTMYieldModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Model loaded successfully.")

# Run evaluation
print("\nRunning evaluation on test set...")
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

# Print results
print("\n" + "="*60)
print("MODEL EVALUATION RESULTS")
print("="*60)
print(f"Test samples evaluated: {len(all_targets)}")
print(f"\nMean Absolute Error (MAE):     {mae:.6f}")
print(f"Root Mean Square Error (RMSE): {rmse:.6f}")
print("="*60)

# Create results directory
os.makedirs("results", exist_ok=True)

# ============================================================================
# Save evaluation metrics to CSV
# ============================================================================
print("\nSaving evaluation results to files...")

# Save MAE and RMSE to CSV
metrics_file = "results/evaluation_metrics.csv"
with open(metrics_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Metric', 'Value'])
    writer.writerow(['Mean Absolute Error (MAE)', f'{mae:.6f}'])
    writer.writerow(['Root Mean Square Error (RMSE)', f'{rmse:.6f}'])
    writer.writerow(['Test Samples Evaluated', len(all_targets)])
print(f"  ✓ Saved: {metrics_file}")

# Save predicted vs actual yield values to CSV
predictions_file = "results/predicted_vs_actual_yield.csv"
with open(predictions_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Sample_Index', 'Actual_Yield', 'Predicted_Yield', 'Error'])
    for i in range(len(all_targets)):
        error = all_predictions[i] - all_targets[i]
        writer.writerow([i, f'{all_targets[i]:.6f}', f'{all_predictions[i]:.6f}', f'{error:.6f}'])
print(f"  ✓ Saved: {predictions_file}")

# Compute prediction errors
errors = all_predictions - all_targets

# ============================================================================
# Plot 1: Actual vs Predicted Yield (Scatter Plot)
# ============================================================================
print("\nGenerating evaluation plots...")

plt.figure(figsize=(10, 8))
plt.scatter(all_targets, all_predictions, alpha=0.6, s=50)
plt.plot([all_targets.min(), all_targets.max()], 
         [all_targets.min(), all_targets.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Yield', fontsize=12)
plt.ylabel('Predicted Yield', fontsize=12)
plt.title('Actual vs Predicted Yield', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/actual_vs_predicted_scatter.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: results/actual_vs_predicted_scatter.png")
plt.show()

# ============================================================================
# Plot 2: Prediction Error Distribution (Histogram)
# ============================================================================
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
plt.axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2, 
            label=f'Mean Error: {np.mean(errors):.4f}')
plt.xlabel('Prediction Error (Predicted - Actual)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Prediction Error Distribution', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('results/error_distribution_histogram.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: results/error_distribution_histogram.png")
plt.show()

# ============================================================================
# Plot 3: Actual vs Predicted Yield Line Plot (Sample Cases)
# ============================================================================
# Select sample cases (first 20 samples for visualization)
num_samples = min(20, len(all_targets))
sample_indices = np.arange(num_samples)

plt.figure(figsize=(12, 6))
plt.plot(sample_indices, all_targets[:num_samples], 'o-', label='Actual Yield', 
         linewidth=2, markersize=6, color='blue')
plt.plot(sample_indices, all_predictions[:num_samples], 's-', label='Predicted Yield', 
         linewidth=2, markersize=6, color='red', alpha=0.7)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Yield', fontsize=12)
plt.title(f'Actual vs Predicted Yield (First {num_samples} Samples)', 
          fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/actual_vs_predicted_lineplot.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: results/actual_vs_predicted_lineplot.png")
plt.show()

print("\n✅ All evaluation plots generated and saved to 'results/' folder.")

