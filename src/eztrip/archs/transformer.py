import torch
import torch.nn as nn
from eztrip.archs.multi_head_attention import MultiHeadAttention
from eztrip.archs.feed_forward import FeedForward

class Decoder(nn.Module):
    def __init__(self, embedding_dim, attention_size,
                 n_heads, attention_dropout=0.1,
                 mh_dropout=0.1, ff_dropout=0.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(embedding_dim, attention_size,
                                            n_heads, attention_dropout,
                                            mh_dropout)
        self.norm1 = torch.nn.LayerNorm(embedding_dim)
        self.norm2 = torch.nn.LayerNorm(embedding_dim)
        self.ff = FeedForward(embedding_dim, dropout_p=ff_dropout)

    def forward(self, x, mask=None):
        norm_x = self.norm1(x)
        x = x + self.self_attn(norm_x, norm_x, norm_x, mask=mask)
        x = x + self.ff(self.norm2(x))
        return x
