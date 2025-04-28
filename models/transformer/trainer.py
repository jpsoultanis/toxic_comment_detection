import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.transformer.model import FullTransformer

class TransformerTrainer:
    def __init__(
        self,
        model: FullTransformer,
        pad_idx: int,
        lr: float = 1e-4,
        step_size: int = 5,
        gamma: float = 0.95,
        device: torch.device = torch.device('cpu'),
    ):
        self.device = device
        self.model = model.to(device)
        self.pad_idx = pad_idx
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # decline learning rate by gamma every step_size epochs
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def train_step(self, src: torch.Tensor, tgt: torch.Tensor) -> float:
        self.model.train()

        src = src.to(self.device, dtype=torch.long)
        tgt = tgt.to(self.device, dtype=torch.long)

        # build padding masks: (batch, seq_len)
        src_pad_mask = (src == self.pad_idx)
        tgt_pad_mask = (tgt == self.pad_idx)

        # decoder input excludes last token
        dec_in = tgt[:, :-1]

        logits = self.model(
            src,
            dec_in,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask[:, :-1],
            memory_key_padding_mask=src_pad_mask,
        )

        loss = self.criterion(
            logits.view(-1, logits.size(-1)),
            tgt[:, 1:].reshape(-1)
        )

        # run backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total = 0.0

        with torch.no_grad():
            for src, tgt in val_loader:
                src = src.to(self.device, dtype=torch.long)
                tgt = tgt.to(self.device, dtype=torch.long)
                src_pad_mask = (src == self.pad_idx)
                tgt_pad_mask = (tgt == self.pad_idx)

                logits = self.model(
                    src,
                    tgt[:, :-1],
                    src_key_padding_mask=src_pad_mask,
                    tgt_key_padding_mask=tgt_pad_mask[:, :-1],
                    memory_key_padding_mask=src_pad_mask,
                )

                total += self.criterion(
                    logits.view(-1, logits.size(-1)),
                    tgt[:, 1:].reshape(-1)
                ).item()

        return total / len(val_loader)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        print("Starting training...")
        for epoch in range(1, epochs + 1):
            train_loss = sum(self.train_step(s, t) for s, t in train_loader) / len(train_loader)
            val_loss = self.evaluate(val_loader)
            self.scheduler.step()
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")