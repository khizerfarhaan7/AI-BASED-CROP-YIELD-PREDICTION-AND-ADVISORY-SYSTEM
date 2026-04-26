import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMYieldModel(nn.Module):
    """
    LSTM model for yield prediction with variable-length sequence support.
    
    Logic:
    - Handles sequences of different lengths (weeks 8-16)
    - Uses pack_padded_sequence to mask future timesteps efficiently
    - Takes last valid hidden state from packed sequence
    - Predicts final yield from partial sequence
    """
    def __init__(self, input_features=2, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x, lengths=None):
        """
        Forward pass with variable-length sequence support.
        
        Args:
            x: Padded sequences of shape (batch, max_seq_len, features)
            lengths: Actual sequence lengths for each sample in batch
        
        Returns:
            Predicted yield values (batch_size,)
        """
        if lengths is not None:
            # Pack sequences to mask padding timesteps
            # This ensures LSTM only processes valid timesteps
            packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_out, (hidden, cell) = self.lstm(packed_x)
            
            # Extract last hidden state from each sequence
            # hidden shape: (num_layers, batch, hidden_size)
            last_hidden = hidden[-1]  # Take last layer's hidden state
        else:
            # Fallback for fixed-length sequences
            lstm_out, (hidden, cell) = self.lstm(x)
            last_hidden = hidden[-1]
        
        # Predict yield from final hidden state
        output = self.fc(last_hidden)
        return output.squeeze(-1)

