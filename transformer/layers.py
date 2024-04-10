"""
Layers used to construct and train a transformer. Includes:

Base layers:
    - Linear
    - Dropout
    - Embedding
    - LayerNorm
    - PositionalEncoding
    - Softmax
    - GELU
    - Sequential

Compound layers:
    - MaskedMultiheadAttention
    - FullyConnected

Decoder sublayers:
    - AttentionSublayer
    - FullyConnectedSublayer

Decoder block:
    - Decoder

Decoder only transformer:
    - Transformer

Cost function:
    - CrossEntropyLoss
"""

import torch
import math
from torch import nn, Tensor

# -------------------------- Base layers -------------------------- #

class Linear(nn.Module):
    """
    Linear layer analogous to nn.Linear.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool=True):
        super().__init__()
        k = 1 / math.sqrt(in_features)
        min, max = -k, k
        w = min + (max - min) * torch.rand(out_features, in_features)
        self.weight = nn.Parameter(w)
        if bias:
            b = min + (max - min) * torch.rand(out_features)
            self.bias = nn.Parameter(b)
        else:
            self.bias = None
        
    def forward(self, x: Tensor) -> Tensor:
        if self.bias is not None:
            return x @ self.weight.T + self.bias
        else:
            return x @ self.weight.T

class Dropout(nn.Module):
    """
    Dropout layer analogous to nn.Dropout.
    """

    def __init__(self, p: float=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        # only apply dropout in training mode
        if self.training:
            mask = torch.rand(x.shape) < self.p
            x[mask] = 0
            return x / (1 - self.p)

        return x
    
class Embedding(nn.Module):
    """
    Embedding layer analogous to nn.Embedding.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(num_embeddings, embedding_dim))

    def forward(self, x: torch.LongTensor) -> Tensor:
        assert x.dtype == torch.int64
        return self.weight[x]

class LayerNorm(nn.Module):
    """
    Layer normalization layer analogous to nn.LayerNorm.
    """

    def __init__(self, normalized_shape, eps: float=1e-5, 
                 elementwise_affine: bool=True, bias: bool=True):
        super().__init__()

        if isinstance(normalized_shape, tuple):
            self.normalized_shape = normalized_shape
        elif isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape, )
        else:
            raise TypeError(
                f'normalized_shape must be int or tuple '
                f'but got {type(normalized_shape)}'
            )

        self.eps = eps

        if elementwise_affine and bias:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        elif elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = None
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        ndim = len(self.normalized_shape) # num dims to normalize
        dims = torch.arange(x.ndim - ndim, x.ndim).tolist() # dims to normalize

        mean = x.mean(dims, keepdim=True)
        var = x.var(dims, unbiased=False, keepdim=True)
        
        if self.weight is not None and self.bias is not None:
            y = (x - mean)*self.weight / torch.sqrt(var + self.eps) + self.bias
        elif self.weight is not None:
            y = (x - mean)*self.weight / torch.sqrt(var + self.eps)
        else:
            y = (x - mean) / torch.sqrt(var + self.eps)

        return y

class PositionalEncoding(nn.Module):
    """
    Positional encoding layer. 

    Adds the matrix PE to the input tensor. PE is of shape (seq_len, d_model)
    and contains the following matrix elements:

    even features: PE(pos, 2i) = sin(pos / 10000**(2i / d_model))
    odd features: PE(pos, 2i + 1) = cos(pos / 10000**(2i / d_model))
    """

    def __init__(self, d_model: int, max_words: int=1000, dropout: float=0.0):
        super().__init__()

        self.d_model = d_model
        self.max_words = max_words
        self.dropout = Dropout(dropout)

        pos = torch.arange(max_words) # word positions
        idx_even = torch.arange(0, d_model, 2) # even word vector features

        pe = torch.empty(max_words, d_model) # positional encoding matrix

        arg = pos[:, None] / 10000**(idx_even / d_model) # argument to sin / cos
        pe[:, 0::2] = torch.sin(arg) # even features
        pe[:, 1::2] = torch.cos(arg) # odd features 

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:

        assert x.size(-2) <= self.max_words, (
            f'length of sequence ({x.size(-2)}) must be less than '
            f'max_words in PE layer ({self.max_words})'
        )
        assert x.size(-1) == self.d_model, (
            f'number of features ({x.size(-1)}) '
            f'must equal d_model ({self.d_model})'
        )

        return self.dropout(x + self.pe[:x.size(-2)])

class Softmax(nn.Module):
    """
    Softmax layer analogous to nn.Softmax.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        x_max = torch.max(x, dim=self.dim, keepdim=True)[0] # for num stability
        exp = torch.exp(x - x_max)
        sum_exp = torch.sum(exp, dim=self.dim, keepdim=True)

        return exp / sum_exp
    
class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) layer. Uses approximate version of GELU 
    function, analogous to nn.GELU(approximate='tanh').
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * x * (
            1 + torch.tanh(math.sqrt(2 / torch.pi) * (x + 0.044715 * x**3))
        )
        
class Sequential(nn.Module):
    """
    Container for storing sequence of layers, analogous to nn.Sequential.
    """
    def __init__(self, *args):
        super().__init__()
        for i, arg in enumerate(args):
            setattr(self, str(i), arg)

    def __getitem__(self, idx):
        return getattr(self, str(idx))
    
    def __iter__(self):
        for module in self._modules.values():
            yield module
    
    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)

        return x

# -------------------------- Compound layers -------------------------- #
    
class MaskedMultiheadAttention(nn.Module):
    """
    Masked multi-head attention layer with optional post-softmax dropout.

    input shape:  (batch_len, seq_len, d_model) or (seq_len, d_model)
    output shape: (batch_len, seq_len, d_model) or (seq_len, d_model)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float=0.0):
        assert d_model % num_heads == 0
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_size = d_model // num_heads
        
        self.wq = nn.Parameter(torch.rand(num_heads, d_model, self.head_size))
        self.wk = nn.Parameter(torch.rand(num_heads, d_model, self.head_size))
        self.wv = nn.Parameter(torch.rand(num_heads, d_model, self.head_size))
        self.wo = nn.Parameter(torch.rand(d_model, d_model))

    def forward(self, x: Tensor) -> Tensor:
        
        # number of tokens in each sequence
        seq_len = x.size(-2)
        
        # create mask with 0 on and below main diag, -inf above
        mask = torch.full((seq_len, seq_len), -torch.inf).triu(1)

        # add new "head" dimension to x
        x.unsqueeze_(-3)

        # linear projections
        q = x @ self.wq
        k = x @ self.wk
        v = x @ self.wv

        # scaled and masked attention scores
        scores = q @ k.transpose(-1, -2)
        scores /= math.sqrt(self.head_size)
        scores += mask

        # attention weights
        weights = Softmax(-1)(scores)

        # dropout
        if self.dropout > 0:
            weights = Dropout(self.dropout)(weights)

        # attention output (weighted sum of value vectors)
        attn_output = weights @ v

        # concatenate heads
        attn_output.transpose_(-2, -3) # transpose head and sequence dimensions
        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1) # concat

        # apply output linear layer
        attn_output = attn_output @ self.wo

        return attn_output, weights
    
class FullyConnected(nn.Module):
    """
    Fully connected layer. Consists of:
        - Linear
        - GeLU
        - Linear

    input shape:  (batch_len, seq_len, d_model) or (seq_len, d_model)
    output shape: (batch_len, seq_len, d_model) or (seq_len, d_model)
    """

    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.layer_stack = Sequential(
            Linear(d_in, d_hidden),
            GELU(),
            Linear(d_hidden, d_in)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layer_stack(x)
    

# -------------------------- Decoder sublayers -------------------------- #

class AttentionSublayer(nn.Module):
    """
    First sublayer in decoder block. Consists of:
        - Layer normalization
        - Masked multi-head attention
        - Dropout
        - Residual connection with input

    input shape:  (batch_len, seq_len, d_model) or (seq_len, d_model)
    output shape: (batch_len, seq_len, d_model) or (seq_len, d_model)

    """

    def __init__(self, d_model: int, num_heads: int, dropout: float=0.0):
        assert d_model % num_heads == 0
        super().__init__()

        # layers
        self.norm = LayerNorm(d_model)
        self.mmha = MaskedMultiheadAttention(d_model, num_heads, dropout)
        self.drop = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm(x)
        y = self.mmha(y)[0]
        y = self.drop(y)
        y += x 

        return y

    def attention_weights(self, x: Tensor) -> Tensor:
        """
        Method for obtaining masked multihead attention weights. 
        
        input shape:  (batch_len, seq_len, d_model) or (seq_len, d_model)
        output shape: (batch_len, num_heads, seq_len, seq_len) or 
            (num_heads, seq_len, seq_len)
        """
        self.eval()
        with torch.no_grad():
            weights = self.mmha(x)[1]
        self.train()

        return weights
    
class FullyConnectedSublayer(nn.Module):
    """
    Second sublayer in decoder block. Consists of:
        - Layer normalization
        - Positionwise fully connected layer
        - Dropout
        - Residual connection with input
    
    input shape:  (batch_len, seq_len, d_model) or (seq_len, d_model)
    output shape: (batch_len, seq_len, d_model) or (seq_len, d_model)
        
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float=0.0):
        super().__init__()

        # layers
        self.norm = LayerNorm(d_model)
        self.ffnn = FullyConnected(d_model, d_ff)
        self.drop = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm(x)
        y = self.ffnn(y)
        y = self.drop(y)
        y += x

        return y


# -------------------------- Decoder block -------------------------- #

class Decoder(nn.Module):
    """
    A decoder block. Consists of:
        - AttentionSublayer
        - FullyConnectedSublayer

    input shape:  (batch_len, seq_len, d_model) or (seq_len, d_model)
    output shape: (batch_len, seq_len, d_model) or (seq_len, d_model)
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, 
                 dropout: float=0.0):
        super().__init__()

        # layers
        self.attn_sublayer = AttentionSublayer(d_model, num_heads, dropout)
        self.ffnn_sublayer = FullyConnectedSublayer(d_model, d_ff, dropout)

    def forward(self, x: Tensor) -> Tensor:
        y = self.attn_sublayer(x)
        y = self.ffnn_sublayer(y)

        return y
    
    def attention_weights(self, x: Tensor) -> Tensor:
        """
        Method for obtaining masked multihead attention weights. Given input x,
        returns attention weights in the specified attention head, with respect
        to specified query position.

        input shape:  (batch_len, seq_len, d_model) or (seq_len, d_model)
        output shape: (batch_len, num_heads, seq_len, seq_len) or 
            (num_heads, seq_len, seq_len)
        """
        return self.attn_sublayer.attention_weights(x)


# -------------------------- Transformer -------------------------- #

class Transformer(nn.Module):
    """
    A decoder only transformer. Consists of the following, in sequence:
        - Embedding layer
        - PositionalEncoding
        - Decoder stack
        - Layer normalization
        - Linear (de-embedding) layer
    """

    def __init__(self, vocab: int, d_model: int, num_heads: int, 
                 num_stacks: int, d_ff: int, dropout: float=0.0):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_stacks = num_stacks
        self.d_ff = d_ff
        self.dropout = dropout

        # layers
        self.embed = Embedding(vocab, d_model)
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        self.decoder_stack = Sequential(*[
            Decoder(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_stacks)])
        self.norm = LayerNorm(d_model)
        self.de_embed = Linear(d_model, vocab, bias=False)
        self.softmax = Softmax(-1) # for inference

        # layer stack
        self.layer_stack = Sequential(
            self.embed,
            self.pe,
            self.decoder_stack,
            self.norm,
            self.de_embed
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layer_stack(x) 
    
    def greedy_output(self, x: Tensor) -> Tensor:
        """
        Method for computing the greedy output of the model. Given input x, the
        method passes x through the tranformer, computes the softmax of the 
        logits, and returns the token with maximal probability at each position.
        """
        self.eval()
        with torch.no_grad():
            probs = self.softmax(self(x))
            output = torch.max(probs, dim=-1)[1]
        self.train()

        return output

    def attention_weights(self, x: Tensor, stack: int) -> Tensor:
        """
        Method for obtaining masked multihead attention weights. Given input x,
        returns attention weights in the specified decoder stack.

        input x: integer tensor of shape (seq_len)
        output: tensor of shape (num_heads, seq_len, seq_len)
        """
        assert stack <= self.num_stacks, (
            f'stack ({stack}) cannot be larger than '
            f'num_stacks ({self.num_stacks})'
        )

        self.eval()
        with torch.no_grad():
            # propagate x through to desired stack
            x = self.embed(x)
            x = self.pe(x)
            x = Sequential(*[self.decoder_stack[i] for i in range(stack)])(x)

            # compute attention weights
            weights = self.decoder_stack[stack].attention_weights(x)

        self.train()

        return weights


# -------------------------- Cost functions -------------------------- #

class CrossEntropyLoss(nn.Module):
    """
    Cross entropy loss, analogous to nn.CrossEntropyLoss.

    Takes input and target and produces output. Input should be logits, 
    target should be class indices, output is always a scalar.
    
    Possible shapes (if swap_dims = False):

    - input (C), target () -> output ()
    - input (B, C), target (B) -> output ()
    - input (B, C, d1, ...), target (B, d1, ...) -> output ()

    Possible shapes (if swap_dims = True):

    - input (C), target () -> output ()
    - input (B, C), target (B) -> output ()
    - input (B, d1, ..., C), target (B, d1, ...) -> output ()

    where B = batch size, C = number of classes.
    """

    def __init__(self, ignore_index: float=-100, swap_dims: bool=False):
        super().__init__()
        self.ignore_index = ignore_index
        self.swap_dims = swap_dims

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # x = input, y = target
        if self.swap_dims:
            assert x.ndim >= 3
            x = x.permute(0, *range(2, x.ndim), 1)
        
        # add batch dim
        if x.ndim == 1:
            assert y.ndim == 0
            x.unsqueeze_(0)
            y.unsqueeze_(0)

        # non-ignore indices 
        idxs = (y != self.ignore_index)

        # z_n := x_{n, y_n}
        grid = torch.meshgrid(*map(torch.arange, y.shape), indexing='ij')
        z = x[grid[0], torch.clamp(y, 0, x.size(1) - 1), *grid[1:]]

        # l_n = -z_n + log(sum_c(exp(x_{n,c})))
        l = -z + x.exp().sum(1).log()

        # if y_n = ignore_index, set l_n = 0
        l[~idxs] = 0

        # cross entropy
        ce = l.sum() / idxs.sum()

        return ce