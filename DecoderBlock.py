import torch
import torch.nn as nn
from SelfAttention import *
from TransformerBlock import TransformerBlock


class DecoderBlock(nn.Module):
    def __init__(self, embedding_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embedding_size)
        self.attention = Attention(embedding_size, heads= heads)
        self.transformer_block = TransformerBlock(
            embedding_size,
            heads,
            dropout,
            forward_expansion
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x,x,x,trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out