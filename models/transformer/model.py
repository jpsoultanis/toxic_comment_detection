import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


# Credit: PositionalEncoding solution inspired by: https://medium.com/%40bavalpreetsinghh/transformer-from-scratch-using-pytorch-28a5d1b2e033
# Note: This class definition is kept for reference, but the model below uses learned positional embeddings.
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

        # Original shape was (max_len, 1, d_model). For batch_first=True, we might need (1, max_len, d_model)
        # Or adjust usage in forward pass if using batch_first=True models.
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: (1, max_len, d_model) for batch_first=True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model) if batch_first=True
        # Adjust slicing for batch_first: self.pe[:, :x.size(1)]
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
    

# Credit to ChatGPT got high level discussion on implementation of attention pooling
class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Linear(d_model, 1)
        
    def forward(self, x, mask):
        attention_scores = self.attention(x).squeeze(-1) # (batch, seq_len)
        
        # apply mask, setting padded positions to large negative value
        attention_scores.masked_fill_(mask, -1e9)
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(1) # (batch, 1, seq_len)
        
        # apply attention weights to input
        pooled = torch.bmm(attention_weights, x).squeeze(1)
        return pooled


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        src_glove_weights: torch.Tensor,
        pad_idx: int,
        num_classes: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        freeze_embeddings: bool = True,
        pooling_type: str = 'attention', # 'mean' or 'attention'
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.pooling_type = pooling_type

        # source token embeddings from GloVe, freeze them if specified
        self.src_embed = nn.Embedding.from_pretrained(
            embeddings=src_glove_weights,
            freeze=freeze_embeddings,
            padding_idx=pad_idx
        )

        # learned positional embeddings
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.pos_drop  = nn.Dropout(dropout)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
        )

        # attention pooling
        if pooling_type == 'attention':
            self.attention_pooling = AttentionPooling(d_model)

        # classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, src_len = src.size()

        # embeddings / positional encodings
        src_emb = self.src_embed(src) * math.sqrt(self.d_model)   # (batch, src_len, d_model)
        src_pos = torch.arange(src_len, device=src.device).unsqueeze(0).expand(batch_size, src_len) # (batch, src_len)
        src_pe = src_emb + self.pos_embed(src_pos) # Add positional embeddings

        # Note: self.pos_drop is only active during training, no effect in eval mode here
        src_pe = self.pos_drop(src_pe)

        memory = self.transformer_encoder(
            src=src_pe,
            src_key_padding_mask=src_key_padding_mask,
        )

        if self.pooling_type == 'attention':
            # attention pooling
            pooled_output = self.attention_pooling(memory, src_key_padding_mask)
        else:
            # mean pooling
            valid_token_mask = ~src_key_padding_mask.unsqueeze(-1).expand_as(memory) # (batch, src_len, d_model)
            summed_output = (memory * valid_token_mask).sum(dim=1)
            valid_token_count = valid_token_mask.sum(dim=1)
            num_valid_tokens = valid_token_count[:, 0].float().clamp(min=1).unsqueeze(1) # (batch, 1)
            pooled_output = summed_output / num_valid_tokens

        logits = self.classifier(pooled_output) # (batch, num_classes)

        return logits