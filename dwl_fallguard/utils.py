import torch
import torch.nn as nn
from pathlib import Path
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_model(model: nn.Module, path: Path):
    """Save model state dict"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(cls, path: Path, **kwargs):
    """Load model from checkpoint"""
    device = kwargs.pop('device', 'cpu')
    model = cls(**kwargs).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model