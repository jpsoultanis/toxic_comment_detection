import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from models.transformer.model import TransformerClassifier

class TransformerTrainer:
    def __init__(
        self,
        model: TransformerClassifier,
        pad_idx: int,
        criterion: nn.Module,
        lr: float = 1e-4,
        clip_grad_norm: float = 1.0,
        device: torch.device = torch.device('cpu'),
    ):
        self.device = device
        self.model = model.to(device)
        self.pad_idx = pad_idx
        self.criterion = criterion

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.clip_grad_norm = clip_grad_norm

        # try to maximize F1 score
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=2,
        )

    # Updated to take src, labels
    def train_step(self, src: torch.Tensor, labels: torch.Tensor) -> float:
        self.model.train()

        src = src.to(self.device, dtype=torch.long)

        # Labels need to be float for BCEWithLogitsLoss/FocalLoss
        labels = labels.to(self.device, dtype=torch.float)

        # build padding mask for source: (batch, seq_len)
        src_pad_mask = (src == self.pad_idx)

        logits = self.model(
            src,
            src_key_padding_mask=src_pad_mask,
        ) 

        # compute loss
        loss = self.criterion(
            logits,
            labels 
        )

        # run backprop
        self.optimizer.zero_grad()
        loss.backward()

        # apply gradient clipping
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)

        self.optimizer.step()
        return loss.item()


    def evaluate(self, val_loader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for src, labels in tqdm(val_loader, desc="Validating"):
                # move to device
                src = src.to(self.device, dtype=torch.long)
                labels = labels.to(self.device, dtype=torch.float)

                # build padding mask
                src_pad_mask = (src == self.pad_idx)

                # forward pass
                logits = self.model(
                    src,
                    src_key_padding_mask=src_pad_mask,
                )  # shape: (batch, num_classes)

                # compute batch loss
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                # collect predictions & labels for metrics
                probs = torch.sigmoid(logits).cpu()
                all_preds.append(probs)
                all_labels.append(labels.cpu())

        # compute averages and F1
        avg_loss = total_loss / len(val_loader)
        preds = torch.cat(all_preds).numpy()
        true = torch.cat(all_labels).numpy()
        preds_binary = (preds > 0.5).astype(int)
        f1 = f1_score(true, preds_binary, average='macro')

        return avg_loss, f1

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        print("Starting training...")
        for epoch in range(1, epochs + 1):
            train_loss = sum(self.train_step(s, lbl) for s, lbl in train_loader) / len(train_loader)
            val_loss, f1_score = self.evaluate(val_loader)
            self.scheduler.step(f1_score) # Step the scheduler and maximize F1 score

            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {f1_score:.4f}, LR = {self.scheduler.get_last_lr()[0]:.6f}")
