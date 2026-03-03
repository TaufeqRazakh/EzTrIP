import torch
import torch.nn as nn
from eztrip.attention import Attention

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, attention_size,
                 n_heads, attention_dropout=0.1,
                 mh_dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList([Attention(embedding_dim, attention_size//n_heads,attention_dropout)
                                             for _ in range(n_heads)])
        self.dropout = nn.Dropout(mh_dropout)
        self.linear = nn.Linear(attention_size, embedding_dim, bias=False)

    def forward(self, k, q, v, mask=None):
        output = []
        for head in self.heads:
            output.append(head(k,q,v,mask))
        output = torch.cat(output, dim=-1)
        return self.dropout(self.linear(output))
