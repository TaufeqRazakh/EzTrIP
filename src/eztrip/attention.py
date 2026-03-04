import torch
import torch.nn as nn
from torch.nn.functional import softmax

class Attention(nn.Module):
    def __init__(self, embedding_dim, attention_size, attention_dropout=0.1):
        super().__init__()
        self.wk = nn.Linear(embedding_dim, attention_size, bias=False)
        self.wq = nn.Linear(embedding_dim, attention_size, bias=False)
        self.wv = nn.Linear(embedding_dim, attention_size, bias=False)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, k, q, v, mask=None):
        """
        k, q, v here are the inputs to the linear layers wk, wq, wv.
        If the input data is the same for all the transformations,
        supply the same input tensor for all 3 arguments.

        input k | q | v [batch_size, seq_length, embedding_dim]
        """
        k = self.wk(k) # [batch_size, seq_length, attention_size]
        q = self.wq(q)
        v = self.wv(v)
        scores = torch.bmm(q, k.transpose(1,2))
        if mask is not None:
            scores = scores.masked_fill(mask, -float('inf'))
        scores = scores / (k.shape[-1] ** 0.5)
        attention = torch.bmm(self.dropout(softmax(scores, dim=-1)), v)
        return attention