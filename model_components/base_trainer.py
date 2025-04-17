import wandb
import json
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import os
import shutil
import torch.optim as optim

class BaseTrainer(ABC):
    """
    Base trainer class for time series forecasting models.
    Maintains original functionality from BaseTrainer but optimized for forecasting tasks.
    
    Key Features:
    1. Time series-specific metric tracking (MSE, MAE, RMSE)
    2. Prediction visualization capabilities
    3. Mixed precision training support
    4. Gradient accumulation
    5. Learning rate scheduling
    6. Checkpoint management
    7. WandB integration
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: dict,
        run_name: str,
        config_file: str,
        device: Optional[str] = None
    ):
        # Device setup
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = model.to(self.device)
        self.config = config

        # Training state
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.training_history = []
        self.use_wandb = config['training'].get('use_wandb', False)
        print(self.use_wandb)
        
        # Optimization setup
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.amp.GradScaler(enabled=self.config['training'].get('mixed_precision', False))
        
        # Experiment setup
        self.expt_root, self.checkpoint_dir, self.viz_dir, self.best_model_path, \
        self.last_model_path = self._init_experiment(run_name, config_file)
        
        

    def _init_experiment(self, run_name: str, config_file: str):
        """Initialize time series forecasting experiment structure"""
        expt_root = Path(os.getcwd()) / 'expts' / run_name
        expt_root.mkdir(parents=True, exist_ok=True)

        # Save config
        shutil.copy2(config_file, expt_root / "config.yaml")

        # Create subdirectories
        checkpoint_dir = expt_root / 'checkpoints'
        viz_dir = expt_root / 'visualizations'
        
        checkpoint_dir.mkdir(exist_ok=True)
        viz_dir.mkdir(exist_ok=True)

        # Initialize WandB
        if self.use_wandb:
            wandb.init(
                project=self.config['training'].get('wandb_project', 'time-series-forecasting'),
                config=self.config,
                name=run_name
            )

        return expt_root, checkpoint_dir, viz_dir, checkpoint_dir / 'best_model.pth', checkpoint_dir / 'last_model.pth'

    @abstractmethod
    def _train_epoch(self, dataloader) -> Dict[str, float]:
        """Implement training logic for one epoch"""
        pass

    @abstractmethod
    def _validate_epoch(self, dataloader) -> Dict[str, float]:
        """Implement validation logic for one epoch"""
        pass

    @abstractmethod
    def train(self, train_loader, val_loader, epochs: int):
        """Full training loop implementation"""
        pass

    @abstractmethod
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluation loop implementation"""
        pass

    def _log_metrics(self, metrics: Dict[str, Dict[str, float]], epoch: int):
        """Enhanced logging with time series-specific visualization"""
        # Update training history
        self.training_history.append({
            'epoch': epoch,
            **metrics,
            'lr': self.optimizer.param_groups[0]['lr']
        })

        # WandB logging
        if self.use_wandb:
            wandb_metrics = {}
            for split, split_metrics in metrics.items():
                for metric_name, value in split_metrics.items():
                    wandb_metrics[f'{split}/{metric_name}'] = value
            wandb_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
            wandb.log(wandb_metrics, step=epoch)

        # Console logging
        print(f"\n📈 Epoch {epoch} Metrics:")
        for split in metrics:
            print(f"  {split.upper():<10}", end="")
            for metric, value in metrics[split].items():
                print(f" | {metric}: {value:.4f}", end="")
            print()

    def _save_forecast_plot(self, predictions: np.ndarray, targets: np.ndarray, epoch: int):
        """Save visualization of forecasts vs actuals"""
        plt.figure(figsize=(12, 6))
        plt.plot(targets, label='Actual')
        plt.plot(predictions, label='Predicted', linestyle='--')
        plt.title(f"Forecast vs Actual - Epoch {epoch}")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        
        plot_path = self.viz_dir / f"forecast_epoch_{epoch}.png"
        plt.savefig(plot_path)
        plt.close()
        
        if self.use_wandb:
            wandb.log({"forecast": wandb.Image(str(plot_path))}, step=epoch)

    def save_checkpoint(self, filename: str):
        """Save training state with time series model specifics"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config,
            'training_history': self.training_history
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)
        
        if self.use_wandb:
            wandb.save(str(self.checkpoint_dir / filename))

    def load_checkpoint(self, filename: str):
        """Load checkpoint with time series model compatibility checks"""
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
        self.training_history = checkpoint['training_history']
        
        # Load mixed precision scaler
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    def cleanup(self):
        """Cleanup resources"""
        if self.use_wandb:
            wandb.finish()
