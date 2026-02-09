"""
MLOps Utilities for Deep Learning with PyTorch
===============================================

Helper functions for Themes 1-8:
- Training loops
- Validation
- Drift detection
- Benchmarking
- Logging

Usage:
    from utils import train_epoch, validate, detect_drift, benchmark_model
"""

import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm
from scipy.stats import ks_2samp
import csv
from datetime import datetime


# ============================================================================
# THEME 2: TRAINING & VALIDATION
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """
    Train model for one epoch
    
    Args:
        model: PyTorch model
        loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: 'cuda' or 'cpu'
    
    Returns:
        epoch_loss: Average loss for epoch
        epoch_acc: Accuracy (%) for epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """
    Validate model
    
    Args:
        model: PyTorch model
        loader: DataLoader for validation/test data
        criterion: Loss function
        device: 'cuda' or 'cpu'
    
    Returns:
        val_loss: Average validation loss
        val_acc: Validation accuracy (%)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validation', leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def save_checkpoint(model, optimizer, epoch, val_acc, filename='checkpoint.pth'):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }
    torch.save(checkpoint, filename)
    print(f"✅ Checkpoint saved: {filename}")


def load_checkpoint(model, optimizer, filename='checkpoint.pth'):
    """Load model checkpoint"""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_acc = checkpoint['val_acc']
    print(f"✅ Checkpoint loaded: epoch {epoch}, val_acc {val_acc:.2f}%")
    return epoch, val_acc


# ============================================================================
# THEME 3: MIXED PRECISION TRAINING
# ============================================================================

def train_epoch_amp(model, loader, criterion, optimizer, scaler, device):
    """
    Train with Automatic Mixed Precision (FP16)
    
    Args:
        model: PyTorch model
        loader: DataLoader
        criterion: Loss function
        optimizer: Optimizer
        scaler: GradScaler for mixed precision
        device: 'cuda' or 'cpu'
    
    Returns:
        epoch_loss, epoch_acc
    """
    from torch.cuda.amp import autocast
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc='Training (AMP)'):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward with autocast
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backward with scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total


# ============================================================================
# THEME 4: BENCHMARKING
# ============================================================================

def benchmark_model(model, input_shape, device='cuda', num_runs=100, warmup=10):
    """
    Benchmark model inference speed
    
    Args:
        model: PyTorch model
        input_shape: Tuple (batch_size, channels, height, width)
        device: 'cuda' or 'cpu'
        num_runs: Number of benchmark iterations
        warmup: Number of warmup iterations
    
    Returns:
        dict with latency_p50, latency_p95, throughput
    """
    model.eval()
    model.to(device)
    
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - start)
    
    times_ms = np.array(times) * 1000  # Convert to ms
    
    return {
        'latency_p50': np.percentile(times_ms, 50),
        'latency_p95': np.percentile(times_ms, 95),
        'latency_mean': np.mean(times_ms),
        'throughput': 1000 / np.mean(times_ms)  # images/sec
    }


def benchmark_onnx(onnx_path, input_shape, num_runs=100):
    """
    Benchmark ONNX model
    
    Args:
        onnx_path: Path to ONNX model
        input_shape: Tuple (batch_size, channels, height, width)
        num_runs: Number of benchmark iterations
    
    Returns:
        dict with latency_p50, latency_p95, throughput
    """
    import onnxruntime as ort
    
    session = ort.InferenceSession(
        onnx_path, 
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    input_name = session.get_inputs()[0].name
    
    # Warmup
    for _ in range(10):
        _ = session.run(None, {input_name: dummy_input})
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = session.run(None, {input_name: dummy_input})
        times.append(time.time() - start)
    
    times_ms = np.array(times) * 1000
    
    return {
        'latency_p50': np.percentile(times_ms, 50),
        'latency_p95': np.percentile(times_ms, 95),
        'throughput': 1000 / np.mean(times_ms)
    }


# ============================================================================
# THEME 5: PRODUCTION LOGGING
# ============================================================================

class InferenceLogger:
    """Log production predictions to CSV"""
    
    def __init__(self, log_file='production_logs.csv'):
        self.log_file = log_file
        # Initialize CSV header
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'pred_class', 'confidence', 'latency_ms'])
    
    def log(self, pred_class, confidence, latency_ms):
        """Log single prediction"""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                pred_class,
                confidence,
                latency_ms
            ])
    
    def log_batch(self, preds, confidences, latencies):
        """Log batch of predictions"""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            timestamp = datetime.now().isoformat()
            for pred, conf, lat in zip(preds, confidences, latencies):
                writer.writerow([timestamp, pred, conf, lat])


# ============================================================================
# THEME 6: DRIFT DETECTION
# ============================================================================

def detect_pixel_drift(baseline_loader, production_loader, num_samples=10000):
    """
    Detect drift using Kolmogorov-Smirnov test on pixel distributions
    
    Args:
        baseline_loader: DataLoader for baseline data
        production_loader: DataLoader for production data
        num_samples: Number of pixels to sample
    
    Returns:
        drift_detected: bool (True if drift detected)
        p_value: p-value from KS test
    """
    baseline_pixels = []
    production_pixels = []
    
    # Collect baseline pixels
    for images, _ in baseline_loader:
        baseline_pixels.extend(images.flatten().numpy())
        if len(baseline_pixels) >= num_samples:
            break
    
    # Collect production pixels
    for images, _ in production_loader:
        production_pixels.extend(images.flatten().numpy())
        if len(production_pixels) >= num_samples:
            break
    
    baseline_pixels = np.array(baseline_pixels[:num_samples])
    production_pixels = np.array(production_pixels[:num_samples])
    
    # KS test
    statistic, p_value = ks_2samp(baseline_pixels, production_pixels)
    
    # Drift if p < 0.05 (5% significance level)
    drift_detected = p_value < 0.05
    
    return drift_detected, p_value


def compute_mmd(X, Y, kernel='rbf', gamma=1.0):
    """
    Maximum Mean Discrepancy between two distributions
    
    Args:
        X: baseline embeddings (N, D) torch.Tensor
        Y: production embeddings (M, D) torch.Tensor
        kernel: 'rbf' or 'linear'
        gamma: RBF kernel bandwidth
    
    Returns:
        mmd_score: float (higher = more drift)
    """
    if kernel == 'rbf':
        XX = torch.exp(-gamma * torch.cdist(X, X) ** 2).mean()
        YY = torch.exp(-gamma * torch.cdist(Y, Y) ** 2).mean()
        XY = torch.exp(-gamma * torch.cdist(X, Y) ** 2).mean()
    else:  # linear
        XX = torch.mm(X, X.t()).mean()
        YY = torch.mm(Y, Y.t()).mean()
        XY = torch.mm(X, Y.t()).mean()
    
    mmd = XX + YY - 2 * XY
    return mmd.item()


def extract_embeddings(model, loader, device, layer_name='avgpool'):
    """
    Extract embeddings from intermediate layer
    
    Args:
        model: PyTorch model
        loader: DataLoader
        device: 'cuda' or 'cpu'
        layer_name: Name of layer to extract from
    
    Returns:
        embeddings: torch.Tensor (N, D)
        labels: torch.Tensor (N,)
    """
    model.eval()
    embeddings = []
    all_labels = []
    
    # Register hook
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # Find layer
    for name, module in model.named_modules():
        if layer_name in name:
            module.register_forward_hook(get_activation(name))
            break
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Extracting embeddings'):
            images = images.to(device)
            _ = model(images)
            
            # Get embeddings
            emb = activation[layer_name]
            if len(emb.shape) > 2:  # If spatial dimensions exist
                emb = emb.mean(dim=[-2, -1])  # Global average pooling
            
            embeddings.append(emb.cpu())
            all_labels.append(labels)
    
    embeddings = torch.cat(embeddings)
    all_labels = torch.cat(all_labels)
    
    return embeddings, all_labels


def detect_embedding_drift(model, baseline_loader, production_loader, device, threshold=0.3):
    """
    Detect drift using MMD on embeddings
    
    Args:
        model: PyTorch model
        baseline_loader: DataLoader for baseline
        production_loader: DataLoader for production
        device: 'cuda' or 'cpu'
        threshold: MMD threshold (>threshold = drift)
    
    Returns:
        drift_detected: bool
        mmd_score: float
    """
    # Extract embeddings
    baseline_emb, _ = extract_embeddings(model, baseline_loader, device)
    production_emb, _ = extract_embeddings(model, production_loader, device)
    
    # Compute MMD
    mmd_score = compute_mmd(baseline_emb, production_emb)
    
    drift_detected = mmd_score > threshold
    
    return drift_detected, mmd_score


# ============================================================================
# THEME 7: RETRAINING UTILITIES
# ============================================================================

def validate_retrained_model(new_model, current_model, baseline_loader, 
                             drifted_loader, criterion, device):
    """
    Validation gate for retrained model
    
    Args:
        new_model: Newly retrained model
        current_model: Current production model
        baseline_loader: Original test set
        drifted_loader: Drifted test set
        criterion: Loss function
        device: 'cuda' or 'cpu'
    
    Returns:
        approved: bool (True if all checks pass)
        metrics: dict with all metrics
    """
    # Evaluate both models
    _, new_acc_baseline = validate(new_model, baseline_loader, criterion, device)
    _, new_acc_drifted = validate(new_model, drifted_loader, criterion, device)
    
    _, curr_acc_baseline = validate(current_model, baseline_loader, criterion, device)
    _, curr_acc_drifted = validate(current_model, drifted_loader, criterion, device)
    
    # Validation checks
    checks = {
        'baseline_maintained': new_acc_baseline >= curr_acc_baseline - 1.0,  # Allow 1% drop
        'drift_improved': new_acc_drifted > curr_acc_drifted,
    }
    
    metrics = {
        'new_acc_baseline': new_acc_baseline,
        'new_acc_drifted': new_acc_drifted,
        'current_acc_baseline': curr_acc_baseline,
        'current_acc_drifted': curr_acc_drifted,
        'checks': checks,
        'approved': all(checks.values())
    }
    
    return metrics['approved'], metrics


# ============================================================================
# UTILITIES
# ============================================================================

def count_parameters(model):
    """Count trainable parameters in model"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_training_summary(history):
    """Print training history summary"""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total epochs: {len(history['train_loss'])}")
    print(f"Best train acc: {max(history['train_acc']):.2f}%")
    print(f"Best val acc: {max(history['val_acc']):.2f}%")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print("="*60)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("MLOps Utilities loaded successfully!")
    print("\nAvailable functions:")
    print("  - train_epoch()")
    print("  - validate()")
    print("  - train_epoch_amp()")
    print("  - benchmark_model()")
    print("  - benchmark_onnx()")
    print("  - InferenceLogger")
    print("  - detect_pixel_drift()")
    print("  - detect_embedding_drift()")
    print("  - compute_mmd()")
    print("  - validate_retrained_model()")
    print("  - set_seed()")
    print("  - count_parameters()")
