import torch
from eztrip.archs.feed_forward import FeedForward

def test_feed_forward_output_shape():
    """
    Tests if the FeedForward module produces an output with the correct shape.
    """
    batch_size = 4
    seq_len = 10
    embedding_dim = 20
    dropout_p = 0.1

    x = torch.randn(batch_size, seq_len, embedding_dim)

    feed_forward_layer = FeedForward(embedding_dim=embedding_dim, dropout_p=dropout_p)

    output = feed_forward_layer(x)

    expected_shape = (batch_size, seq_len, embedding_dim)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"
