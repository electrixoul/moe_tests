import torch
from torch.utils.data import Dataset
import numpy as np


class CharDataset(Dataset):
    """Character-level dataset for language modeling."""
    
    def __init__(self, data_path, seq_len=256, split='train', train_split=0.9, seed=42):
        """
        Args:
            data_path: Path to text file
            seq_len: Sequence length
            split: 'train' or 'val'
            train_split: Fraction of data for training
            seed: Random seed for reproducibility
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Build vocabulary
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        # Encode text
        data = np.array([self.char_to_idx[ch] for ch in text], dtype=np.int64)
        
        # Split data
        n = len(data)
        split_idx = int(n * train_split)
        
        if split == 'train':
            self.data = data[:split_idx]
        else:
            self.data = data[split_idx:]
        
        self.seq_len = seq_len
        
    def __len__(self):
        # Return number of possible sequences
        return max(0, len(self.data) - self.seq_len)
    
    def __getitem__(self, idx):
        # Get sequence and target (shifted by 1)
        chunk = self.data[idx:idx + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].copy()).long()
        y = torch.from_numpy(chunk[1:].copy()).long()
        return x, y
    
    def decode(self, indices):
        """Decode indices to text."""
        return ''.join([self.idx_to_char[i] for i in indices])
    
    def encode(self, text):
        """Encode text to indices."""
        return [self.char_to_idx[ch] for ch in text]
