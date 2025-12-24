import os
import torch


def _ensure_dir(path: str):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def save_checkpoint(model, path: str):
    _ensure_dir(path)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path: str, device):
    if not os.path.exists(path):
        return False
    model.load_state_dict(torch.load(path, map_location=device), strict=False)
    return True


def save_history(model, history_dir: str, step: int, prefix: str = "candidate"):
    os.makedirs(history_dir, exist_ok=True)
    path = os.path.join(history_dir, f"{prefix}_step_{step}.pth")
    torch.save(model.state_dict(), path)
    return path
