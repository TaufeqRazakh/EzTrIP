import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, embedding_dim, dropout_p):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, embedding_dim*4)
        self.dropout = nn.Dropout(dropout_p)
        self.linear2 = nn.Linear(embedding_dim*4, embedding_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.dropout(self.activation(self.linear1(x)))
        return self.linear2(x)