import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from utils.utils import AverageMeter

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, logger, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config
        self.logger = logger
        self.device = device
        
        # Class Imbalance Handling
        self.criterion = self._get_loss_function()
        
        # Optimizer & Scheduler
        self.optimizer = optim.SGD(
            model.parameters(), 
            lr=config.LEARNING_RATE, 
            momentum=0.9, 
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=config.LR_MILESTONES, gamma=config.LR_GAMMA
        )
        
        self.best_acc = 0.0
        self.patience_counter = 0
        self.class_names = ["Safe", "Warning", "Critical"] # Adjust if you have different names

    def _get_loss_function(self):
        # Calculate weights dynamically from training data
        all_labels = []
        for _, y in self.train_loader:
            all_labels.extend(y.numpy())
        
        counts = torch.bincount(torch.tensor(all_labels))
        weights = 1.0 / (counts.float() + 1e-6)
        weights = weights / weights.sum()
        self.logger.info(f"Computed Class Weights: {weights.tolist()}")
        
        return nn.CrossEntropyLoss(
            weight=weights.to(self.device), 
            label_smoothing=self.cfg.LABEL_SMOOTHING
        )

    def train_epoch(self):
        self.model.train()
        losses = AverageMeter()
        accs = AverageMeter()
        
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            _, pred = torch.max(out, 1)
            acc = (pred == y).float().mean() * 100.0
            losses.update(loss.item(), x.size(0))
            accs.update(acc.item(), x.size(0))
            
        self.scheduler.step()
        return losses.avg, accs.avg

    def validate(self, epoch_num, run_dir):
        """
        Modified validation loop to calculate Precision, Recall, F1 and save Confusion Matrix.
        """
        self.model.eval()
        losses = AverageMeter()
        accs = AverageMeter()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = self.criterion(out, y)
                
                _, pred = torch.max(out, 1)
                acc = (pred == y).float().mean() * 100.0
                
                losses.update(loss.item(), x.size(0))
                accs.update(acc.item(), x.size(0))
                
                # Collect for sklearn metrics
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                
        # --- Advanced Metrics Calculation ---
        # weighted average handles class imbalance
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        # --- Confusion Matrix Generation (Only if this is the best model so far to save space) ---
        # We return the raw metrics, but we can save the plot here if needed
        self.last_cm_preds = all_preds
        self.last_cm_labels = all_labels
        
        return losses.avg, accs.avg, precision, recall, f1

    def save_confusion_matrix(self, preds, labels, run_dir, epoch):
        """Generates and saves a confusion matrix image"""
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"confusion_matrix_epoch_{epoch}.png"))
        plt.close()

    def fit(self, run_dir):
        csv_path = os.path.join(run_dir, "metrics.csv")
        # Added P/R/F1 to CSV headers
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "Precision", "Recall", "F1"])

            for epoch in range(self.cfg.EPOCHS):
                t0 = time.time()
                t_loss, t_acc = self.train_epoch()
                # Unpack new metrics
                v_loss, v_acc, val_p, val_r, val_f1 = self.validate(epoch+1, run_dir)
                dt = time.time() - t0
                
                self.logger.info(f"Epoch {epoch+1:02d} | {dt:.1f}s | "
                                 f"Train: {t_loss:.4f} ({t_acc:.1f}%) | "
                                 f"Val: {v_loss:.4f} ({v_acc:.1f}%) | "
                                 f"F1: {val_f1:.4f}")
                
                # Log to CSV
                writer.writerow([epoch+1, t_loss, t_acc, v_loss, v_acc, val_p, val_r, val_f1])
                f.flush()

                # Checkpoint & Early Stopping
                if v_acc > self.best_acc:
                    self.best_acc = v_acc
                    self.patience_counter = 0
                    
                    # Save Model
                    torch.save(self.model.state_dict(), os.path.join(run_dir, "best_model.pth"))
                    torch.save(self.model.state_dict(), self.cfg.MODEL_SAVE_NAME) 
                    
                    # Save Confusion Matrix for this "Best" epoch
                    self.save_confusion_matrix(self.last_cm_preds, self.last_cm_labels, run_dir, epoch+1)
                    
                    self.logger.info(f"★ New Best Model Saved! ({v_acc:.2f}%) - Confusion Matrix Saved")
                # else:
                #     self.patience_counter += 1
                    
                # if self.patience_counter >= self.cfg.EARLY_STOPPING_PATIENCE:
                #     self.logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                #     break