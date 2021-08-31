import torch
import torch.nn as nn
from TransformerBlock import TransformerBlock                

class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embedding_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_lenght,
    ):
        super(Encoder, self).__init__()
        self.embedding_size = embedding_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(max_lenght, embedding_size)
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
                
            ]
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        N, seq_length = x.shape
        position = torch.arrange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(position))
        )
        
        for layers in self.layers:
            out = layer(out, out, out, mask)
            
        return out