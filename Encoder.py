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
        