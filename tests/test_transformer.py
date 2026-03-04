import torch
import torch.nn as nn
from eztrip.transformer import Decoder

def test_decoder_output_shape():
    """
    Tests if the Decoder module produces an output with the correct shape.
    """
    batch_size = 2
    seq_len = 8
    embedding_dim = 32
    attention_size = 20
    n_heads = 4
    attention_dropout = 0.1
    mh_dropout = 0.1
    ff_dropout = 0.1

    x = torch.randn(batch_size, seq_len, embedding_dim)

    decoder_layer = Decoder(embedding_dim, attention_size,
                            n_heads, attention_dropout,
                            mh_dropout, ff_dropout)

    output = decoder_layer(x)

    expected_shape = (batch_size, seq_len, embedding_dim)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"

    # Test with mask
    mask = torch.ones(seq_len, seq_len).bool().tril(diagonal=0)
    output_with_mask = decoder_layer(x, mask=mask)
    assert output_with_mask.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output_with_mask.shape}"
