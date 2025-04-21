import math
import torch
import torch.nn as nn
import torch.optim as optim

# Credit: PositionalEncoding solution inspired by: https://medium.com/%40bavalpreetsinghh/transformer-from-scratch-using-pytorch-28a5d1b2e033
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Positional Encoding for Transformer models. This module adds positional encodings to input embeddings
        to give the model information about the position of each token in the sequence.

        The positional encodings are computed using sine and cosine functions of different frequencies.

        Args:
            d_model: the dimension of the model
            dropout: the dropout probability
            max_len: the maximum length of the input sequences
        """
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerTrainer:
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        lr=1e-4,
        step_size=5,
        gamma=0.95,
        device='cpu'
    ):
        self.device = device
        self.pos_encoder = PositionalEncoding(d_model, dropout).to(device)

        # full Transformer model
        self.model = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        ).to(device)

        # token embeddings and output generator
        self.src_embedding = nn.Embedding(src_vocab_size, d_model).to(device)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model).to(device)
        self.linear = nn.Linear(d_model, tgt_vocab_size).to(device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # From the docs: "Decays the learning rate of each parameter group by gamma every step_size epochs"
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=step_size,
            gamma=gamma
        )

    def _generate_mask(self, sz):
        """
        Generate a causal mask for sequence-to-sequence decoding. This prevents 
        the decoder from attending to future tokens in the sequence.
        """
        # Credit: mark generation solution inspired by https://medium.com/@swarms/understanding-masking-in-pytorch-for-attention-mechanisms-e725059fd49f
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def train_step(self, src, tgt):
        """
        src: (S, N)   source sequence
        tgt: (T, N)   target sequence (with <sos> and <eos>)
        """
        self.model.train()

        # prepare mask
        seq_len = tgt.size(0)
        tgt_mask = self._generate_mask(seq_len).to(self.device)

        # embeddings + positional encoding
        src_emb = self.pos_encoder(self.src_embedding(src))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt[:-1]))

        # forward pass
        output = self.model(src_emb, tgt_emb, tgt_mask=tgt_mask)
        logits = self.linear(output)

        # compute loss
        loss = self.criterion(
            logits.view(-1, logits.size(-1)),
            tgt[1:].reshape(-1)
        )

        # backprop & update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for src, tgt in val_loader:
                src = src.to(self.device)
                tgt = tgt.to(self.device)

                src_emb = self.pos_encoder(self.src_embedding(src))
                tgt_emb = self.pos_encoder(self.tgt_embedding(tgt[:-1]))
                tgt_mask = self._generate_mask(
                    tgt.size(0) - 1
                ).to(self.device)

                output = self.model(src_emb, tgt_emb, tgt_mask=tgt_mask)
                logits = self.linear(output)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    tgt[1:].reshape(-1)
                )
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, train_loader: torch.DataLoader, val_loader: torch.DataLoader, epochs: int):
        for epoch in range(1, epochs + 1):
            train_loss = 0.0
            for src, tgt in train_loader:
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                train_loss += self.train_step(src, tgt)

            val_loss = self.evaluate(val_loader)
            self.scheduler.step()

            # print(
            #     f"Epoch {epoch}: "
            #     f"Train Loss={train_loss/len(train_loader):.4f} | "
            #     f"Val Loss={val_loss:.4f}"
            # )