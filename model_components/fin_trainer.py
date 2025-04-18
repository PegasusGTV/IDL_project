from .base_trainer import BaseTrainer
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchmetrics

class TimeSeriesForecastingTrainer(BaseTrainer):
    def __init__(self, model, config, run_name, config_file, device=None):
        super().__init__(model, config, run_name, config_file, device)
        self.loss_fn = nn.MSELoss()
        self.mae_metric = torchmetrics.MeanAbsoluteError().to(device)
        self.forecast_horizon = model.forecast_horizon

    def _calculate_accuracy(self, predictions, targets, threshold=0.1):
        """
        Calculate percentage accuracy based on a threshold.
        Args:
            predictions: Predicted values (torch.Tensor).
            targets: Ground truth values (torch.Tensor).
            threshold: Maximum allowable error as a fraction of the target value.
        Returns:
            Percentage accuracy (float).
        """
        relative_error = torch.abs(predictions - targets) / torch.abs(targets)
        accurate_predictions = (relative_error <= threshold).float()
        accuracy = accurate_predictions.mean().item() * 100  # Convert to percentage
        return accuracy

    def _train_epoch(self, dataloader: DataLoader):
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        batch_bar = tqdm(total=len(dataloader), desc="Training")
        
        for batch_idx, (src, tgt) in enumerate(dataloader):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            self.optimizer.zero_grad()
            
            with torch.autocast(device_type=self.device, dtype=torch.float16):
                predictions = self.model(src)
                loss = self.loss_fn(predictions, tgt)
            
            # Gradient accumulation
            loss = loss / self.config['training']['gradient_accumulation_steps']
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item() * src.size(0)
            self.mae_metric.update(predictions, tgt)
            
            # Calculate accuracy for the batch
            batch_accuracy = self._calculate_accuracy(predictions, tgt)
            total_accuracy += batch_accuracy * src.size(0)
            
            batch_bar.set_postfix(
                loss=f"{total_loss/(batch_idx+1):.4f}",
                mae=f"{self.mae_metric.compute().item():.4f}",
                accuracy=f"{(total_accuracy / ((batch_idx + 1) * src.size(0))):.2f}%"
            )
            batch_bar.update()
        
        batch_bar.close()
        return {
            'train_loss': total_loss / len(dataloader.dataset),
            'train_mae': self.mae_metric.compute().item(),
            'train_accuracy': total_accuracy / len(dataloader.dataset)
        }

    def _validate_epoch(self, dataloader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        mae_metric = torchmetrics.MeanAbsoluteError().to(self.device)
        
        with torch.no_grad():
            for src, tgt in dataloader:
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                
                predictions = self.model(src)
                loss = self.loss_fn(predictions, tgt)
                
                total_loss += loss.item() * src.size(0)
                mae_metric.update(predictions, tgt)
        
         # Calculate accuracy for the batch
            batch_accuracy = self._calculate_accuracy(predictions, tgt)
            total_accuracy += batch_accuracy * src.size(0)
    
        return {
            'val_loss': total_loss / len(dataloader.dataset),
            'val_mae': mae_metric.compute().item(),
            'val_accuracy': total_accuracy / len(dataloader.dataset)
        }

    def train(self, train_loader, val_loader, epochs: int):
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            train_metrics = self._train_epoch(train_loader)
            val_metrics = self._validate_epoch(val_loader)
            
            # Update learning rate scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['val_loss'])
            else:
                self.scheduler.step()
            
            # Log metrics
            metrics = {
                'epoch': epoch+1,
                'train': train_metrics,
                'val': val_metrics
            }
            self._log_metrics(metrics, epoch)
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                self.save_checkpoint('best_model.pth')
                
            self.save_checkpoint('last_model.pth')
            

    def evaluate(self, test_loader):
        return self._validate_epoch(test_loader)
