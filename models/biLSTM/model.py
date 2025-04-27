# model.py
import torch
import torch.nn as nn

class ToxicCommentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, hidden_size=64, output_dim=6,dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_dim, hidden_size,num_layers=3,bidirectional=True,batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size*2, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        h_forward = h_n[0]  # forward direction
        h_backward = h_n[1]  # backward direction
        h_cat = torch.cat((h_forward, h_backward), dim=1)  # [batch, hidden*2]
        out = self.fc(self.dropout(h_cat))
        return out
