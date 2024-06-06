import torch
import torch.nn as nn
import torch.nn.functional as F

hyperparams = {
    "dropout":0.6, # dropout prob
    "block_dim":8, # number of blocks used to predict for next time-step
    "embd_dim":64, # embedding dimension for the attention mechanism
    "n_head":4, # number of self-attention heads in multi-head attention block
    "n_layer": 4 # number of multi-head attention blocks
}

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self):
        super().__init__()
        embd_dim = hyperparams['embd_dim']
        dropout = hyperparams['dropout']
        
        self.net = nn.Sequential(
            nn.Linear(embd_dim, 4 * embd_dim),
            nn.ReLU(),
            nn.Linear(4 * embd_dim, embd_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        embd_dim = hyperparams['embd_dim']
        block_size = hyperparams['block_dim']
        dropout = hyperparams['dropout']
        self.key = nn.Linear(embd_dim, head_size, bias=False)
        self.query = nn.Linear(embd_dim, head_size, bias=False)
        self.value = nn.Linear(embd_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill( self.tril[:T, :T] == 0, float('-inf') ) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, head_size):
        super().__init__()
        num_heads = hyperparams['n_head']
        embd_dim = hyperparams['embd_dim']
        dropout = hyperparams['dropout']
        
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embd_dim, embd_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        embd_dim = hyperparams['embd_dim']
        n_head = hyperparams['n_head']
        head_size = embd_dim // n_head
        self.sa = MultiHeadAttention(head_size)
        self.ffwd = FeedFoward()
        self.ln1 = nn.LayerNorm(embd_dim)
        self.ln2 = nn.LayerNorm(embd_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Transformer_model(nn.Module):
    # def __init__(self, data_dim, block_dim = 8, embd_dim = 64, n_head = 4, n_layer = 4):
    def __init__(self, data_dim):
        super().__init__()
        block_dim = hyperparams['block_dim']
        embd_dim = hyperparams['embd_dim']
        n_head = hyperparams['n_head']
        n_layer = hyperparams['n_layer']        
        self.position_embedding_table = nn.Embedding(block_dim, embd_dim)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(embd_dim) # final layer norm
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder =  nn.Linear(data_dim, embd_dim)
        self.decoder = nn.Linear(embd_dim, data_dim)
        
    def forward(self, idx):
        B, T, C = idx.shape
        # idx and targets are both (B,T,1920) tensors
        # x = idx #(B, T, 1920) 
        # need to make it (B, T, 64) before doing other things
        x = self.encoder(idx)
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T, 64)
        x += pos_emb # (B,T, 64)
        x = self.blocks(x) # (B,T, 64)
        x = self.ln_f(x) # (B,T, 64)
        logits = self.decoder(x) # (B,T, 1920)
        return logits
        