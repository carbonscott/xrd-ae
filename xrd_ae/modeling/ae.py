import torch
import torch.nn as nn

import math
from einops import rearrange

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, use_flash=False):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.use_flash = use_flash

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        b, n, d = x.shape
        h = self.heads

        # Project to q, k, v
        qkv = rearrange(self.to_qkv(x), 'b n (three h d) -> three b h n d', three=3, h=h)
        q, k, v = qkv

        if self.use_flash:
            # Flash attention implementation
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        else:
            # Regular attention
            dots = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dim_head)
            attn = dots.softmax(dim=-1)
            attn_output = torch.matmul(attn, v)

        # Merge heads and project
        out = rearrange(attn_output, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, use_flash=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, dim_head, use_flash=use_flash)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4, bias=False),
            nn.GELU(),
            nn.Linear(dim * 4, dim, bias=False)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads=8, dim_head=64, use_flash=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, dim_head, use_flash=use_flash) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads=8, dim_head=64, use_flash=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, dim_head, use_flash=False) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ViTAutoencoder(nn.Module):
    def __init__(
        self,
        image_size=(1920, 1920),
        patch_size=128,
        latent_dim=256,
        dim=1024,
        depth=2,
        heads=8,
        dim_head=64,
        use_flash=True,
        norm_pix=True  # Flag for patch normalization
    ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.norm_pix = norm_pix

        # Calculate patches
        self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        patch_dim = 1 * patch_size * patch_size

        # Encoder components
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_dim, dim, bias=True),
            nn.LayerNorm(dim)
        )

        ## # Learnable position embeddings
        ## self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        # Fixed positional embedding
        pos_embed = self._get_sinusoidal_pos_embed(self.num_patches, dim)
        self.register_buffer('pos_embedding', pos_embed.unsqueeze(0))

        # Transformer encoder
        self.encoder = nn.ModuleList([
            nn.Sequential(
                TransformerBlock(dim, heads, dim_head, use_flash),
                nn.Dropout(0.1)
            ) for _ in range(depth)
        ])

        # Projection to latent space
        self.to_latent = nn.Sequential(
            nn.LayerNorm(dim * self.num_patches),
            nn.Linear(dim * self.num_patches, latent_dim, bias=True)
        )

        # Decoder components
        self.from_latent = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, dim * self.num_patches, bias=True)
        )

        # Transformer decoder
        self.decoder = nn.ModuleList([
            nn.Sequential(
                TransformerBlock(dim, heads, dim_head, use_flash),
                nn.Dropout(0.1)
            ) for _ in range(depth)
        ])

        # Patch reconstruction
        self.to_pixels = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim, bias=True)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with orthogonal initialization"""
        def init_ortho(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_ortho)

        # Get the last Linear layer from each Sequential
        to_latent_linear = [m for m in self.to_latent if isinstance(m, nn.Linear)][-1]
        from_latent_linear = [m for m in self.from_latent if isinstance(m, nn.Linear)][-1]

        # Initialize decoder projection as transpose of encoder
        with torch.no_grad():
            from_latent_linear.weight.copy_(to_latent_linear.weight.t())

    def _get_sinusoidal_pos_embed(self, num_pos, dim, max_period=10000):
        """
        Generate fixed sinusoidal position embeddings.

        Args:
            num_pos   : Number of positions (patches)
            dim       : Embedding dimension
            max_period: Maximum period for the sinusoidal functions. Controls the
                        range of wavelengths from 2π to max_period⋅2π. Higher values
                        create longer-range position sensitivity.

        Returns:
            torch.Tensor: Position embeddings of shape (num_pos, dim)
        """
        assert dim % 2 == 0, "Embedding dimension must be even"

        # Use half dimension for sin and half for cos
        omega = torch.arange(dim // 2, dtype=torch.float32) / (dim // 2 - 1)
        omega = 1. / (max_period**omega)  # geometric progression of wavelengths

        pos = torch.arange(num_pos, dtype=torch.float32)
        pos = pos.view(-1, 1)  # Shape: (num_pos, 1)
        omega = omega.view(1, -1)  # Shape: (1, dim//2)

        # Now when we multiply, broadcasting will work correctly
        angles = pos * omega  # Shape: (num_pos, dim//2)

        # Compute sin and cos embeddings
        pos_emb_sin = torch.sin(angles)  # Shape: (num_pos, dim//2)
        pos_emb_cos = torch.cos(angles)  # Shape: (num_pos, dim//2)

        # Concatenate to get final embeddings
        pos_emb = torch.cat([pos_emb_sin, pos_emb_cos], dim=1)  # Shape: (num_pos, dim)
        return pos_emb

    def normalize_patches(self, patches):
        """
        Normalize each patch independently
        patches: (B, N, P*P) where N is number of patches, P is patch size
        """
        if not self.norm_pix:
            return patches

        # Calculate mean and var over patch pixels
        mean = patches.mean(dim=-1, keepdim=True)
        var = patches.var(dim=-1, keepdim=True)
        patches = (patches - mean) / (var + 1e-6).sqrt()

        return patches

    def denormalize_patches(self, patches, orig_mean, orig_var):
        """
        Denormalize patches using stored statistics
        """
        if not self.norm_pix:
            return patches

        patches = patches * (orig_var + 1e-6).sqrt() + orig_mean
        return patches

    def patchify(self, x):
        """Convert image to patches"""
        return rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                        p1=self.patch_size, p2=self.patch_size)

    def unpatchify(self, patches):
        """Convert patches back to image"""
        h_patches = w_patches = int(math.sqrt(self.num_patches))
        return rearrange(patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                        h=h_patches, w=w_patches, p1=self.patch_size, p2=self.patch_size)

    def encode(self, x):
        # Convert image to patches
        patches = self.patchify(x)

        # Store original statistics for denormalization if needed
        if self.norm_pix:
            self.orig_mean = patches.mean(dim=-1, keepdim=True)
            self.orig_var = patches.var(dim=-1, keepdim=True)
            patches = self.normalize_patches(patches)

        # Patch embedding
        tokens = self.patch_embed(patches)

        # Add positional embedding
        x = tokens + self.pos_embedding

        # Transformer encoding
        for encoder_block in self.encoder:
            x = x + encoder_block(x)

        # Project to latent space
        latent = self.to_latent(rearrange(x, 'b n d -> b (n d)'))
        return latent

    def decode(self, z):
        # Project from latent space
        x = self.from_latent(z)
        x = rearrange(x, 'b (n d) -> b n d', n=self.num_patches)

        # Transformer decoding
        for decoder_block in self.decoder:
            x = x + decoder_block(x)

        # Reconstruct patches
        patches = self.to_pixels(x)

        # Denormalize if needed
        if self.norm_pix:
            patches = self.denormalize_patches(patches, self.orig_mean, self.orig_var)

        # Convert patches back to image
        return self.unpatchify(patches)

    def forward(self, x):
        latent = self.encode(x)
        return self.decode(latent)
