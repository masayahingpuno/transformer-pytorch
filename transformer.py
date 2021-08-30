import torch
import torch.nn as nn

class SelfAttention(nn.module):
    def __init__(self, embedding_size, heads):
        super(SelfAttention, self).__init()
        self.embedding_size = embedding_size
        self.heads = heads
        self.head_dimension = embedding_size // heads
        
        # Check if the embedding_size is divisible by the heads size
        assert(
            self.head_dimension * heads == embedding_size 
        ), "Embedding size needs to be divisible by the heads size"
        
        self.values = nn.Linear(self.head_dimension, self.head_dimension, bias=False)
        self.keys = nn.Linear(self.head_dimension, self.head_dimension, bias=False)
        self.queries = nn.Linear(self.head_dimension, self.head_dimension, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dimension, embedding_size)