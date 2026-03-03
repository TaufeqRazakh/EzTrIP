import torch
from eztrip.multi_head_attention import MultiHeadAttention

def test_attention_output_shape():
    embedding_dim = 20
    attention_size = 25,
    n_heads = 5

    seq_len = 5
    batch_size = 1
    x = torch.randn(batch_size, seq_len, embedding_dim)
    
    attention_layer = MultiHeadAttention(20, 25, 5)
    output = attention_layer(x, x, x)

    expected_shape = (batch_size, seq_len, embedding_dim)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"
    print(expected_shape)
    print(output.shape)
