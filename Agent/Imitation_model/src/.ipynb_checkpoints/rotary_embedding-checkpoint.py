from math import pi
import torch
from torch import einsum
from torch import nn, Tensor
from einops import rearrange, repeat
from typing import Literal, Optional, Tuple


# Helper function to check if a value exists (is not None)
def exists(val):
    return val is not None


# Helper function to return a default value if val is None
def default(val, d):
    return val if exists(val) else d


# Helper function to concatenate broadcasted tensors along a specified dimension
def broadcat(tensors, dim=-1):
    broadcasted_tensors = torch.broadcast_tensors(*tensors)
    return torch.cat(broadcasted_tensors, dim=dim)


# Helper function to slice a tensor at a specific dimension
def slice_at_dim(t: Tensor, dim_slice: slice, *, dim: int) -> Tensor:
    dim += (t.ndim if dim < 0 else 0)
    colons = [slice(None)] * t.ndim
    colons[dim] = dim_slice
    return t[tuple(colons)]


# Helper function to apply rotary embeddings to the half-dimensions
def rotate_half(x: Tensor) -> Tensor:
    """Rotates the last two dimensions of the input tensor."""
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


# Apply rotary embedding to a tensor
def apply_rotary_emb(
    freqs: Tensor, t: Tensor, start_index: int = 0, scale: float = 1.0, seq_dim: int = -2,
    freqs_seq_dim: Optional[int] = None
) -> Tensor:
    """
    Applies rotary embeddings to the input tensor t using the frequencies in freqs.
    
    Parameters:
    - freqs: Frequencies for rotary embedding.
    - t: Input tensor to apply rotary embedding.
    - start_index: Start index for applying rotary embedding.
    - scale: Scaling factor for rotary embedding.
    - seq_dim: Sequence dimension for applying the embedding.
    - freqs_seq_dim: Sequence dimension for frequencies (optional).
    
    Returns:
    - Tensor: The tensor after applying rotary embedding.
    """
    dtype = t.dtype

    if t.ndim == 3 or exists(freqs_seq_dim):
        freqs_seq_dim = default(freqs_seq_dim, 0)
        seq_len = t.shape[seq_dim]
        freqs = slice_at_dim(freqs, slice(-seq_len, None), dim=freqs_seq_dim)

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is too small to apply rotary embeddings for {rot_dim} positions.'

    # Split tensor into parts: left, middle (to be transformed), and right
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]

    # Apply rotary embeddings
    t_transformed = (t_middle * freqs.cos() * scale) + (rotate_half(t_middle) * freqs.sin() * scale)
    
    # Concatenate left, transformed middle, and right parts
    out = torch.cat((t_left, t_transformed, t_right), dim=-1)

    return out.type(dtype)


# Class for rotary embedding, including support for learned frequencies and extrapolation (xpos)
class RotaryEmbedding(nn.Module):
    def __init__(
        self, dim: int, custom_freqs: Optional[Tensor] = None, freqs_for: Literal['lang', 'pixel', 'constant'] = 'lang',
        theta: float = 10000, max_freq: int = 10, num_freqs: int = 1, learned_freq: bool = False,
        use_xpos: bool = False, xpos_scale_base: int = 512, interpolate_factor: float = 1.0, 
        theta_rescale_factor: float = 1.0, seq_before_head_dim: bool = False, cache_if_possible: bool = True,
        cache_max_seq_len: int = 8192
    ):
        """
        Initialize the Rotary Embedding class.
        
        Parameters:
        - dim: Dimension of the rotary embedding.
        - custom_freqs: Custom frequencies for the embedding (optional).
        - freqs_for: Specifies the type of frequencies ('lang', 'pixel', 'constant').
        - theta: Base frequency for generating rotary embeddings.
        - max_freq: Maximum frequency (for pixel-related embeddings).
        - num_freqs: Number of frequency bands (for constant frequencies).
        - learned_freq: Whether to learn frequencies during training.
        - use_xpos: Whether to use extrapolation for long sequence lengths.
        - xpos_scale_base: Scaling base for xpos extrapolation.
        - interpolate_factor: Factor for frequency interpolation.
        - theta_rescale_factor: Rescaling factor for theta.
        - seq_before_head_dim: Whether the sequence dimension is before the head dimension.
        - cache_if_possible: Whether to cache the computed frequencies.
        - cache_max_seq_len: Maximum sequence length for caching.
        """
        super().__init__()

        # Rescale theta based on NTK-aware scaling
        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()

        self.cache_if_possible = cache_if_possible
        self.cache_max_seq_len = cache_max_seq_len

        # Cached frequency tensors (for long sequences)
        self.register_buffer('cached_freqs', torch.zeros(cache_max_seq_len, dim), persistent=False)
        self.register_buffer('cached_freqs_seq_len', torch.tensor(0), persistent=False)

        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)

        self.learned_freq = learned_freq

        # Dummy tensor for determining device
        self.register_buffer('dummy', torch.tensor(0), persistent=False)

        # Default sequence dimension
        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # Interpolation factor for frequencies
        assert interpolate_factor >= 1.0
        self.interpolate_factor = interpolate_factor

        # Extrapolation (xpos) setup
        self.use_xpos = use_xpos

        if use_xpos:
            scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
            self.scale_base = xpos_scale_base
            self.register_buffer('scale', scale, persistent=False)
            self.register_buffer('cached_scales', torch.zeros(cache_max_seq_len, dim), persistent=False)
            self.register_buffer('cached_scales_seq_len', torch.tensor(0), persistent=False)

        # Static method for applying rotary embedding
        self.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(self):
        return self.dummy.device

    # Helper method to compute sequence positions for xpos
    def get_seq_pos(self, seq_len: int, device, dtype, offset=0) -> Tensor:
        return (torch.arange(seq_len, device=device, dtype=dtype) + offset) / self.interpolate_factor

    # Apply rotary embeddings to queries
    def rotate_queries_or_keys(self, t: Tensor, seq_dim: Optional[int] = None, offset=0, scale=None) -> Tensor:
        """Rotates queries or keys using rotary embedding."""
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert not self.use_xpos or exists(scale), 'You must pass both queries and keys when using xpos.'

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, device=device, dtype=dtype, offset=offset)

        freqs = self.forward(seq, seq_len=seq_len, offset=offset)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')

        return apply_rotary_emb(freqs, t, scale=default(scale, 1.), seq_dim=seq_dim)

    # Forward pass to compute rotary frequencies
    def forward(self, t: Tensor, seq_len: Optional[int] = None, offset: int = 0) -> Tensor:
        """Computes the rotary frequencies based on sequence length."""
        should_cache = (
            self.cache_if_possible and not self.learned_freq and
            exists(seq_len) and self.freqs_for != 'pixel' and
            (offset + seq_len) <= self.cache_max_seq_len
        )

        if should_cache and exists(self.cached_freqs) and (offset + seq_len) <= self.cached_freqs_seq_len.item():
            return self.cached_freqs[offset:(offset + seq_len)].detach()

        freqs = self.freqs
        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)

        if should_cache and offset == 0:
            self.cached_freqs[:seq_len] = freqs.detach()
            self.cached_freqs_seq_len.copy_(seq_len)

        return freqs