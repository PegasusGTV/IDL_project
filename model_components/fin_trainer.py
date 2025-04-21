from .base_trainer import BaseTrainer
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchmetrics
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


class TimeSeriesForecastingTrainer(BaseTrainer):
    def __init__(self, model, config, run_name, config_file, scaler, device=None):
        super().__init__(model, config, run_name, config_file, device)
        self.loss_fn = nn.MSELoss()
        # default reduction- aceraging v/s summing 
        self.mae_metric = torchmetrics.MeanAbsoluteError().to(device)
        self.forecast_horizon = model.forecast_horizon
        self.run_name = run_name
        self.target_scaler = scaler

    def _train_epoch(self, dataloader: DataLoader):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_elements = 0  # For accuracy
        total_samples = 0   # For loss normalization
        batch_bar = tqdm(total=len(dataloader), desc="Training")
        
        for batch_idx, (src, memory, tgt_shifted, tgt_golden) in enumerate(dataloader):
            src = src.to(self.device)
            tgt_shifted = tgt_shifted.to(self.device)
            memory = memory.to(self.device)
            tgt_golden = tgt_golden.to(self.device)

            # last_close = src[:, -1:, 3:4]  # Shape: [B, 1, 1]
            # naive_pred = last_close.repeat(1, self.forecast_horizon, 1)
            # naive_loss = self.loss_fn(naive_pred, tgt_golden)
            # print(naive_loss)
            # print(src.shape)
            # print(tgt.shape)
            
            self.optimizer.zero_grad()
            
            with torch.autocast(device_type=self.device, dtype=torch.float16):
                predictions = self.model(memory, tgt_shifted)
                # print(f"train predictions are {predictions[0]}")
                print(f"train targets are {tgt_golden.shape}")
                print(f"train predictions are {predictions.shape}")
                print(f"train src are {src.shape}")
                print(f"train memory are {memory.shape}")
                loss = self.loss_fn(predictions, tgt_golden)
            
            # Gradient accumulation
            loss = loss / self.config['training']['gradient_accumulation_steps']
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            # Update metrics
            # Track UNADJUSTED loss (pre-division)
            total_loss += loss.item() * self.config['training']['gradient_accumulation_steps'] * src.size(0)
            total_samples += src.size(0)
            
            # Accuracy calculation
            pct_error = torch.abs(predictions - tgt_golden) / (torch.abs(tgt_golden) + 1e-8)
            total_correct += (pct_error <= 0.1).sum().item()
            total_elements += predictions.numel()
            
            # Update progress bar (show current batch's metrics)
            current_batch_loss = loss.item() * self.config['training']['gradient_accumulation_steps']
            current_accuracy = (pct_error <= 0.1).float().mean().item()
            
            batch_bar.set_postfix(
                loss=f"{current_batch_loss:.4f}",  # Show current batch loss
                mae=f"{self.mae_metric.compute().item():.4f}",
                batch_accuracy=f"{100 * current_accuracy:.2f}%"
            )
            batch_bar.update()
        
        batch_bar.close()
        # Final metrics (properly normalized)
        avg_loss = total_loss / total_samples  # Per-sample loss
        avg_accuracy = 100 * total_correct / total_elements

        return {
            'train_loss': avg_loss,
            'train_mae': self.mae_metric.compute().item(),
            'train_accuracy': avg_accuracy
        }

    def _validate_epoch(self, dataloader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        mae_metric = torchmetrics.MeanAbsoluteError().to(self.device)
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for src, memory, tgt_shifted, tgt_golden in dataloader:
                src = src.to(self.device)
                tgt_shifted = tgt_shifted.to(self.device)
                memory = memory.to(self.device)
                tgt_golden = tgt_golden.to(self.device)
                
                predictions = self.model(memory, tgt_shifted)
                # print(f"val predictions are {predictions[0]}")
                # print(f"val targets are {tgt[0]}")
                loss = self.loss_fn(predictions, tgt_golden)
                
                total_loss += loss.item() * src.size(0)
                mae_metric.update(predictions, tgt_golden)

                all_preds.append(predictions.detach().cpu())
                all_targets.append(tgt_golden.detach().cpu())
        
        # Concatenate all batches
        all_preds = torch.cat(all_preds, dim=0).numpy().reshape(-1, 1)
        all_targets = torch.cat(all_targets, dim=0).numpy().reshape(-1, 1)

        y_pred = self.target_scaler.inverse_transform(all_preds)
        y_true = self.target_scaler.inverse_transform(all_targets)

        plt.figure(figsize=(8, 6))
        plt.title("Raw (Normalized) Predictions vs Targets")
        plt.plot(all_targets, color='blue', linestyle='--')
        plt.plot(all_preds, color='red', linestyle='--')
        plt.xlabel("Time step")
        plt.ylabel("Normalized Close Value")
        plt.legend(('actual (normalized)', 'prediction (normalized)'), loc='upper left', fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(8, 8))
        plt.title("actual v/s predicted values")
        plt.plot(y_true, color='b')
        plt.plot(y_pred, color='r')
        plt.xlabel("Time value (day)")
        plt.ylabel("Close value (point)")
        plt.legend(('actual', 'prediction'), loc='upper left', fontsize=12)
        plt.grid(True)
        plt.show()

    
        return {
            'val_loss': total_loss / len(dataloader.dataset),
            'val_mae': mae_metric.compute().item(),
            'val_accuracy': 0 / len(dataloader.dataset)
        }

    def train(self, train_loader, val_loader, epochs: int):
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            train_metrics = self._train_epoch(train_loader)
            # val_metrics = self._validate_epoch(val_loader)
            
            # Update learning rate scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(train_metrics['train_loss'])
            else:
                self.scheduler.step()
            
            # Log metrics
            metrics = {
                'epoch': epoch+1,
                'train': train_metrics,
                # 'val': val_metrics
            }
            self._log_metrics(metrics, epoch)
            
            # Save best model
            # if val_metrics['val_loss'] < best_val_loss:
            #     best_val_loss = val_metrics['val_loss']
            #     self.save_checkpoint('best_model.pth')
                
            self.save_checkpoint('last_model.pth')
            

    def evaluate(self, test_loader):
        return self._validate_epoch(test_loader)
