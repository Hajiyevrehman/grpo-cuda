import torch
import os
import shutil
import random
import numpy as np

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # These settings make cuda deterministic but slower
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cleanup_cache(build_dir=None):
    """Clean up cached files and GPU memory"""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if build_dir and os.path.exists(build_dir):
        shutil.rmtree(build_dir, ignore_errors=True)

def safe_mkdir(directory):
    """Safely create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)