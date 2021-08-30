import torch
import torch.nn as nn
#from einops import rearrange, reduce, repeat

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
        
    def forward (self, values, keys, queries, mask):
        # Get the number of training examples
        N = query.shape[0] # 0 means ROW
        
        value_len, keys_len, query_len = values.shape[1], keys.shape[1], query.shape[1] # 1 means column
        
        # Split the embedding into self.heads different pieces 
        values = values.reshape(N, value_len, self.heads, self.head_dimension)
        keys = keys.reshape(N, keys_len, self.heads, self.head_dimension)
        queries = queries.reshape(N, query_len, self.heads, self.head_dimension)
        
        # Matrix multiplication using einops 
        qk_matmul = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) # Change to einops after
        # keys shape is N, keys_len, self.heads, self.head_dimension
        # queries shape is N, query_len, self.heads, self.head_dimension
        # qk_matmul shape is N, heads, query_len, keys_len
        
        # Mask padded indices so their weights become 0
        if mask is not None:
            qk_matmul = qk_matmul.masked_fil(mask == 0)
            
        # Normalize using softmax and divide by scaling factor
        attention = torch.softmax(qk_matmul / (self.embedding_size ** (1 / 2)), dim=3)
        # attention shape is N, heads, query_len, keys_len
        
        out = torch.einsum("nhql, nlhd -> nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dimension
        )
        
        """
        Attention shape: N, heads, query_len, keys_len
        Value shape: N, value_len, heads, head_dimension
        out shape after matmul: N, query_len, heads, head_dimension
        Then reshape and flatten the self.heads and self.head_dimension by multiplying them
        """
        
        out = self.fc_out(out)
        # Shape: N, query_len, embedding_size
        
        return out