import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# Credit: PositionalEncoding solution inspired by: https://medium.com/%40bavalpreetsinghh/transformer-from-scratch-using-pytorch-28a5d1b2e033
class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer models. This module adds positional encodings to input embeddings
    to give the model information about the position of each token in the sequence.

    The positional encodings are computed using sine and cosine functions of different frequencies.

    Args:
        d_model: the dimension of the model
        dropout: the dropout probability
        max_len: the maximum length of the input sequences
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer('pe', pe.unsqueeze(1))  # (max_len, 1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, d_model)
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class FullTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        pad_idx: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        src: torch.Tensor,    
        tgt: torch.Tensor,   
        src_key_padding_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        src = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        tgt = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))

        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        out = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.generator(out)  # (T, N, vocab_size)
