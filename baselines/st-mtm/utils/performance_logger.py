"""
Performance Logger Utility
Records model parameters, inference time, and GPU memory usage
"""
import torch
import time
import numpy as np
import os
from datetime import datetime


class PerformanceLogger:
    """
    A utility class to track and log model performance metrics including:
    - Model parameters (total and trainable)
    - Inference time
    - GPU memory usage
    """
    
    def __init__(self, model, device, log_dir='./test_results/'):
        """
        Initialize the performance logger
        
        Args:
            model: PyTorch model to monitor
            device: Device the model is on
            log_dir: Directory to save logs
        """
        self.model = model
        self.device = device
        self.log_dir = log_dir
        
        # Calculate model parameters
        self.total_params = sum(p.numel() for p in model.parameters())
        self.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Initialize tracking lists
        self.inference_times = []
        self.gpu_memory_used = []
        self.batch_count = 0
        
    def start_batch(self):
        """Record start of batch inference"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.batch_start_time = time.time()
        
    def end_batch(self):
        """Record end of batch inference"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_time = time.time() - self.batch_start_time
        self.inference_times.append(batch_time)
        
        # Record GPU memory
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated(self.device) / 1024**2  # MB
            self.gpu_memory_used.append(gpu_mem)
            
        self.batch_count += 1
        
    def get_metrics(self):
        """
        Get average metrics
        
        Returns:
            dict: Dictionary containing all performance metrics
        """
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0
        avg_gpu_memory = np.mean(self.gpu_memory_used) if self.gpu_memory_used else 0
        peak_gpu_memory = max(self.gpu_memory_used) if self.gpu_memory_used else 0
        
        metrics = {
            'total_params': self.total_params,
            'trainable_params': self.trainable_params,
            'avg_inference_time': avg_inference_time,
            'avg_gpu_memory': avg_gpu_memory,
            'peak_gpu_memory': peak_gpu_memory,
            'total_batches': self.batch_count
        }
        
        return metrics
    
    def print_metrics(self, model_name='Model', dataset='Unknown'):
        """Print metrics to console"""
        metrics = self.get_metrics()
        
        print("="*80)
        print(f"Performance Metrics - {model_name} on {dataset}")
        print("="*80)
        print(f"Model Parameters:")
        print(f"  Total Parameters: {metrics['total_params']:,}")
        print(f"  Trainable Parameters: {metrics['trainable_params']:,}")
        print(f"\nInference Metrics:")
        print(f"  Average Inference Time: {metrics['avg_inference_time']:.4f}s per batch")
        print(f"  Total Batches: {metrics['total_batches']}")
        
        if torch.cuda.is_available():
            print(f"\nGPU Memory Usage:")
            print(f"  Average GPU Memory: {metrics['avg_gpu_memory']:.2f} MB")
            print(f"  Peak GPU Memory: {metrics['peak_gpu_memory']:.2f} MB")
        print("="*80)
        
    def save_log(self, log_path, setting='', additional_metrics=None):
        """
        Save detailed log to file
        
        Args:
            log_path: Path to save the log file
            setting: Experiment setting string
            additional_metrics: Dict of additional metrics to log (e.g., MSE, MAE)
        """
        metrics = self.get_metrics()
        
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        with open(log_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"Model Performance Metrics\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            if setting:
                f.write(f"Setting: {setting}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("Model Parameters\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Parameters: {metrics['total_params']:,}\n")
            f.write(f"Trainable Parameters: {metrics['trainable_params']:,}\n")
            f.write(f"Model Size: {metrics['total_params'] * 4 / (1024**2):.2f} MB (float32)\n\n")
            
            f.write("-"*80 + "\n")
            f.write("Inference Metrics\n")
            f.write("-"*80 + "\n")
            f.write(f"Average Inference Time: {metrics['avg_inference_time']:.4f}s per batch\n")
            f.write(f"Total Batches: {metrics['total_batches']}\n")
            f.write(f"Throughput: {1/metrics['avg_inference_time'] if metrics['avg_inference_time'] > 0 else 0:.2f} batches/sec\n")
            
            if torch.cuda.is_available():
                f.write(f"\nGPU Memory Usage:\n")
                f.write(f"Average GPU Memory: {metrics['avg_gpu_memory']:.2f} MB\n")
                f.write(f"Peak GPU Memory: {metrics['peak_gpu_memory']:.2f} MB\n")
            
            if additional_metrics:
                f.write("\n")
                f.write("-"*80 + "\n")
                f.write("Prediction Metrics\n")
                f.write("-"*80 + "\n")
                for key, value in additional_metrics.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.6f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
            
            f.write("\n" + "="*80 + "\n")
            
    def append_to_score(self, score_path):
        """
        Append performance metrics to existing score.txt file
        
        Args:
            score_path: Path to the score file
        """
        metrics = self.get_metrics()
        
        with open(score_path, 'a') as f:
            f.write(f"Model Parameters: Total={metrics['total_params']:,}, Trainable={metrics['trainable_params']:,}\n")
            f.write(f"Average Inference Time: {metrics['avg_inference_time']:.4f}s per batch\n")
            if torch.cuda.is_available():
                f.write(f"Average GPU Memory: {metrics['avg_gpu_memory']:.2f} MB\n")
            f.write("\n")


class InferenceTimer:
    """Context manager for timing inference"""
    
    def __init__(self, logger):
        self.logger = logger
        
    def __enter__(self):
        self.logger.start_batch()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.end_batch()
        return False
