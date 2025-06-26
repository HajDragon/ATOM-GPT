"""
Data processing utilities for the nanoGPT project
"""

import os
import pickle
import numpy as np

def load_dataset(data_dir):
    """Load training and validation datasets from binary files"""
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    return train_data, val_data

def load_meta(data_dir):
    """Load metadata (tokenizer info) from the dataset"""
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        return meta
    return None

def get_batch(data, block_size, batch_size, device):
    """Generate a batch of data"""
    ix = np.random.randint(len(data) - block_size, size=(batch_size,))
    x = np.stack([data[i:i+block_size] for i in ix])
    y = np.stack([data[i+1:i+1+block_size] for i in ix])
    
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    return x, y

def encode_text(text, tokenizer):
    """Encode text using the given tokenizer"""
    return tokenizer.encode(text)

def decode_text(tokens, tokenizer):
    """Decode tokens using the given tokenizer"""
    return tokenizer.decode(tokens)
