import torch
from eztrip.attention import Attention

def test_attention_output_shape():
    embedding_dim = 20
    attention_size = 25

    seq_len = 5
    batch_size = 1
    x = torch.randn(batch_size, seq_len, embedding_dim)

    attention_layer = Attention(embedding_dim, attention_size)
    output = attention_layer(x, x, x)

    expected_shape = (batch_size, seq_len, attention_size)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"
    print(expected_shape)
    print(output.shape)
