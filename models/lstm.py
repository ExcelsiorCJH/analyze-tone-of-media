import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.types_ import *


class BiLSTMAttn(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_class: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout_p: float = 0.3,
    ):
        super(BiLSTMAttn, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        num_directs = 1
        if bidirectional:
            num_directs = 2

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.embed.weight)
        self.bilstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_p,
        )
        self.linear = nn.Linear(hidden_dim * num_directs, num_class)
        self.dropout = nn.Dropout(dropout_p)

    def attention_layer(self, query, key):
        query = query.squeeze(0)
        attn_scores = torch.bmm(key, query.unsqueeze(2)).squeeze(2)
        attn_weights = F.softmax(attn_scores, dim=1)
        contexts = torch.bmm(key.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(2)
        return contexts, attn_weights

    def forward(self, sequence):
        x = self.embed(sequence)
        x = self.dropout(x)
        output, _ = self.bilstm(x)
        output, attn_weights = self.attention_layer(output[:, -1, :], output)
        output = self.linear(output)
        return output, attn_weights
