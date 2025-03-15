# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
from torch import nn
import math


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)

class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (0) #(-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)

class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)

class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC

class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC

class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)

from typing import Optional

# https://nn.labml.ai/transformers/mha.html
class PrepareForMultiHeadAttention(nn.Module):
    """
    This module does a linear transformation and splits the vector into given
    number of heads for multi-head attention.
    This is used to transform **key**, **query**, and **value** vectors.
    """

    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()
        # Linear layer for linear transform
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        # Number of heads
        self.heads = heads
        # Number of dimensions in vectors in each head
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        # Input has shape `[seq_len, batch_size, d_model]` or `[batch_size, d_model]`.
        # We apply the linear transformation to the last dimension and split that into
        # the heads.
        x = x.cuda()
        head_shape = x.shape[:-1]
        # Linear transform
        x = self.linear(x)
        # Split last dimension into heads
        x = x.view(*head_shape, self.heads, self.d_k)
        # Output has shape `[seq_len, batch_size, heads, d_k]` or `[batch_size, heads, d_model]`
        return x

class MultiHeadAttention(nn.Module):
    """
    ## Multi-Head Attention Module
    This computes scaled multi-headed attention for given `query`, `key` and `value` vectors.

    In simple terms, it finds keys that matches the query, and gets the values of
     those keys.

    Softmax is calculated along the axis of of the sequence (or time).
    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        """
        * `heads` is the number of heads.
        * `d_model` is the number of features in the `query`, `key` and `value` vectors.
        """
        super().__init__()
        # Number of features per head
        self.d_k = d_model // heads
        # Number of heads
        self.heads = heads

        # These transform the `query`, `key` and `value` vectors for multi-headed attention.
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)

        # Softmax for attention along the time dimension of `key`
        self.softmax = nn.Softmax(dim=1)

        # Output layer
        self.output = nn.Linear(d_model, d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
        # Scaling factor before the softmax
        self.scale = (self.d_k) ** (-0.5)
        # We store attentions so that it can be used for logging, or other computations if needed
        # self.attn = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        ### Calculate scores between queries and keys
        This method can be overridden for other variations like relative attention.
        """
        # Calculate $Q K^\top$ or $S_{ijbh} = \sum_d Q_{ibhd} K_{jbhd}$
        return torch.einsum('ibhd,jbhd->ijbh', query, key)

    def prepare_mask(self, mask: torch.Tensor, query_shape: list[int], key_shape: list[int]):
        """
        `mask` has shape `[seq_len_q, seq_len_k, batch_size]`, where first dimension is the query dimension.
        If the query dimension is equal to $1$ it will be broadcasted.
        """
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]
        # Same mask applied to all heads.
        mask = mask.unsqueeze(-1)
        # resulting mask has shape `[seq_len_q, seq_len_k, batch_size, heads]`
        return mask

    def forward(self, *,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """
        `query`, `key` and `value` are the tensors that store
        collection of *query*, *key* and *value* vectors.
        They have shape `[seq_len, batch_size, d_model]`.

        `mask` has shape `[seq_len, seq_len, batch_size]` and
        `mask[i, j, b]` indicates whether for batch `b`,
        query at position `i` has access to key-value at position `j`.
        """

        # `query`, `key` and `value`  have shape `[seq_len, batch_size, d_model]`
        seq_len, batch_size, _ = query.shape
        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)
        # Prepare `query`, `key` and `value` for attention computation.
        # These will then have shape `[seq_len, batch_size, heads, d_k]`.
        query = self.query(query.cuda())
        key = self.key(key.cuda())
        value = self.value(value.cuda())

        # Compute attention scores $Q K^\top$.
        # This gives a tensor of shape `[seq_len, seq_len, batch_size, heads]`.
        scores = self.get_scores(query, key)

        # Scale scores $\frac{Q K^\top}{\sqrt{d_k}}$
        scores *= self.scale

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # $softmax$ attention along the key sequence dimension
        # $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = self.softmax(scores)

        # Apply dropout
        attn = self.dropout(attn)

        # Multiply by values
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$
        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)

        # Save attentions for any other calculations 
        # self.attn = attn.detach()

        # Concatenate multiple heads
        x = x.reshape(seq_len, batch_size, -1)

        # Output layer
        return self.output(x)

# https://nn.labml.ai/transformers/xl/relative_mha.html
def shift_right(x: torch.Tensor):
    """
    This method shifts $i^{th}$ row of a matrix by $i$ columns.

    If the input is `[[1, 2 ,3], [4, 5 ,6], [7, 8, 9]]`, the shifted
    result would be `[[1, 2 ,3], [0, 4, 5], [6, 0, 7]]`.
    *Ideally we should mask out the lower triangle but it's ok for our purpose*.
    """
    # Concatenate a column of zeros
    x = x.cuda()
    zero_pad = x.new_zeros(x.shape[0], 1, *x.shape[2:])
    x_padded = torch.cat([x, zero_pad], dim=1)

    # Reshape and remove excess elements from the end
    x_padded = x_padded.view(x.shape[1] + 1, x.shape[0], *x.shape[2:])
    x = x_padded[:-1].view_as(x)
    return x

class RelativeMultiHeadAttention(MultiHeadAttention):
    """
    ## Relative Multi-Head Attention Module

    We override [Multi-Head Attention](mha.html) module so we only need to 
    write the `get_scores` method.
    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1):
        # The linear transformations do not need a bias since we
        # explicitly include it when calculating scores.
        # However having a bias for `value` might make sense.
        super().__init__(heads, d_model, dropout_prob, bias=False)

        # Number of relative positions
        self.P = 1024

        # Relative positional embeddings for key relative to the query.
        # We need $2P$ embeddings because the keys can be before or after the query.
        self.key_pos_embeddings = nn.Parameter(torch.zeros((self.P * 2, heads, self.d_k)), requires_grad=True)
        # Relative positional embedding bias for key relative to the query.
        self.key_pos_bias = nn.Parameter(torch.zeros((self.P * 2, heads)), requires_grad=True)
        # Positional embeddings for the query is independent of the position of the query
        self.query_pos_bias = nn.Parameter(torch.zeros((heads, self.d_k)), requires_grad=True)

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        ### Get relative attention scores

        With absolute attention
        """
        key_pos_emb = self.key_pos_embeddings[self.P - key.shape[0]:self.P + query.shape[0]]
        key_pos_bias = self.key_pos_bias[self.P - key.shape[0]:self.P + query.shape[0]]
        query_pos_bias = self.query_pos_bias[None, None, :, :]

        ac = torch.einsum('ibhd,jbhd->ijbh', query + query_pos_bias, key)
        b = torch.einsum('ibhd,jhd->ijbh', query, key_pos_emb)
        d = key_pos_bias[None, :, None, :]
        bd = shift_right(b + d)
        # Remove extra positions
        bd = bd[:, -key.shape[0]:]

        # Return the sum
        return ac + bd    

class ConformerMHSA(nn.Module):
    """ 
    Multi-Headed Self-Attention Module part of the Conformer Encoder as per
    https://arxiv.org/abs/2005.08100

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        num_heads (int): ``num_heads`` to use in nn.MultiheadAttention()
    """
    def __init__(
        self,
        num_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.layer_norm = nn.LayerNorm(num_features)
        self.MHSA = RelativeMultiHeadAttention(heads=num_heads, d_model=num_features, dropout_prob=dropout)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        x = self.layer_norm(x)
        x = self.MHSA(query=x, key=x, value=x)
        x = self.dropout(x)
        x = inputs + x
        
        return x  # (T, N, num_features)

class ConformerConv(nn.Module):
    """ Convolution Module part of the Conformer Encoder as per
    https://arxiv.org/abs/2005.08100

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(
        self,
        channels: int,
        width: int,
        kernel_width: int=17
    ) -> None:
        super().__init__()
        self.channels = channels
        self.width = width
        num_features = channels*width
        self.num_features = num_features

        self.layer_norm = nn.LayerNorm(num_features)
        self.pointwise_conv1 = nn.Conv1d(in_channels=num_features, out_channels=2*num_features, kernel_size=1)
        self.GLU = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(in_channels=self.num_features, out_channels=self.num_features, 
                                        kernel_size=kernel_width, padding='same', groups=self.num_features)
                                        
        self.batch_norm = nn.BatchNorm1d(num_features=num_features)
        self.swish = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(in_channels=num_features, out_channels=num_features, kernel_size=1)
        self.dropout = nn.Dropout()
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        inputs = inputs.cuda()
        x = inputs
        x = self.layer_norm(x)
        x = inputs.movedim(0, -1)
        x = self.pointwise_conv1(x)
        x = self.GLU(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC
        x = inputs + x

        return x # (T, N, num_features)
    
class ConformerFF(nn.Module):
    """ Feed Forward Module part of the Conformer Encoder as per
    https://arxiv.org/abs/2005.08100

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """
    def __init__(self, num_features: int) -> None:
        super().__init__()
        
        self.fc_block = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, num_features),
            nn.SiLU(),
            nn.Dropout(),
            nn.Linear(num_features, num_features),
            nn.Dropout()
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.cuda()
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC
    
from collections.abc import Sequence

class ConformerBlock(nn.Module):
    """Conformer Block part of the Conformer Encoder as per
    https://arxiv.org/abs/2005.08100

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channel (int): integers indicating number of channels`.
        kernel_width (int): The kernel size of the temporal convolutions.
    
    """
    def __init__(self, num_features: int, block_channel: int = 24, kernel_width: int = 31) -> None:
        super().__init__()
        
        self.FF1 = ConformerFF(num_features=num_features)
        self.MHSA = ConformerMHSA(num_features=num_features, num_heads=4)
        self.Conv = ConformerConv(block_channel, num_features // block_channel, kernel_width)
        self.FF2 = ConformerFF(num_features=num_features)
        self.layer_norm = nn.LayerNorm(num_features)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.cuda()
        x = inputs
        x = x + 0.5*self.FF1(x)
        x = x + self.MHSA(x)
        x = x + self.Conv(x)
        x = x + 0.5*self.FF2(x)

        return self.layer_norm(x)

class ConformerEncoder(nn.Module):
    """
    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24),
        kernel_width: Sequence[int] = (25),
    ) -> None:
        super().__init__()
        assert len(block_channels) > 0
        conv_blocks: list[nn.Module] = []
        for i in range(0, len(block_channels)):
            assert (
                num_features % block_channels[i] == 0
            ), "block_channels must evenly divide num_features"
            conv_blocks.extend(
                [
                    ConformerBlock(num_features, block_channels[i], kernel_width[i])
                ]
            )
        self.conformer_blocks = nn.Sequential(*conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.cuda()
        return self.conformer_blocks(inputs)  # (T, N, num_features)

class NormRotationMLP(nn.Module):
    """
    Args:
        num_bands (int)
        electrode_channels (int)
        in_features (int)
        mlp_features (Sequence[int])
    """

    def __init__(
        self,
        num_bands: int,
        electrode_channels: int,
        in_features: int,
        mlp_features: Sequence[int]
    ) -> None:
        super().__init__()

        self.normMLP = nn.Sequential(
            # (T, N, bands=2, C=16, freq)
            SpectrogramNorm(channels=num_bands * electrode_channels),
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=num_bands,
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.normMLP(inputs)

from emg2qwerty.charset import charset
class TDSConvLR(nn.Module):
    """
    Args:
        num_features (int)
        block_channels (Sequence[int]),
        kernel_width (int)

    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int],
        kernel_width: int
    ) -> None:
        super().__init__()

        self.TDS = TDSConvEncoder(
                    num_features=num_features,
                    block_channels=block_channels,
                    kernel_width=kernel_width,
                )
            # (T, N, num_classes)
        self.flatten = nn.Flatten(start_dim=2)
        self.linear = nn.Linear(2*num_features, charset().num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        LR = inputs
        LR = inputs.movedim(2, 0)
        L = LR[0]
        R = LR[1]
        L = self.flatten(L)
        R = self.flatten(R)
        L = self.TDS(L) 
        R = self.TDS(R)
        LR = torch.cat((L, R), dim=-1)
        LR = self.linear(LR)
        return self.softmax(LR)

class TDSConv2dBlock_v2(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
            padding = 'same'
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        
        # skip connection
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC
        x = x + inputs

        # Layer norm over C
        return self.layer_norm(x)  # TNC

class TDSConvEncoder_v2(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock_v2(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)

# MHSA using Relative Positional Embedding
class RelativePositionAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_len: int=1024, dropout: float=0.1, use_mask: bool=True) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        d_head = d_model // num_heads
        self.max_len = max_len
        
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.Er = nn.Parameter(torch.randn(max_len, d_head))

        self.use_mask = use_mask
        
        if use_mask:
            self.register_buffer(
                "mask", 
                torch.tril(torch.ones(max_len, max_len))
                .unsqueeze(0).unsqueeze(0)
            )
        # self.mask: (1, 1, max_len, max_len)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor: 
        # inputs: (N, T, d_model) = (batch size, sequence length, dimension of embedding)
        N, T, _ = inputs.shape

        if T > self.max_len:
            raise ValueError(
                "sequence length exceeds model capacity"
            )

        k_t = self.key(inputs).reshape(N, T, self.num_heads, -1).permute(0, 2, 3, 1)
        # k_t.shape = (N, num_heads, d_head, T)
        v = self.value(inputs).reshape(N, T, self.num_heads, -1).transpose(1, 2)
        q = self.query(inputs).reshape(N, T, self.num_heads, -1).transpose(1, 2)
        # shape = (N, num_heads, T, d_head)

        start = self.max_len - T
        Er_t = self.Er[start:, :].transpose(0, 1)
        # Er_t.shape = (d_head, T)
        QEr = torch.matmul(q, Er_t)
        # QEr.shape = (N, num_heads, T, T)
        Srel = self.skew(QEr)
        # Srel.shape = (N, num_heads, seq_len, seq_len)
        QK_t = torch.matmul(q, k_t)
        # QK_t.shape = (batch_size, num_heads, seq_len, seq_len)

        attn = (QK_t + Srel) / math.sqrt(q.size(-1))


        if self.use_mask:
            mask = self.mask[:, :, :T, :T]
            # mask.shape = (1, 1, T, T)
            attn = attn.masked_fill(mask == 0, float("-inf"))
            # attn.shape = (N, num_heads, T, T)

        attn = nn.functional.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (N, num_heads, T, d_head)
        out = out.transpose(1, 2)
        # out.shape == (N, T, num_heads, d_head)
        out = out.reshape(N, T, -1)
        # out.shape == (N, T, d_model)

        return self.dropout(out)
        
    def skew(self, QEr: torch.Tensor) -> torch.Tensor:
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = nn.functional.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel

class ConvSubsample(nn.Module):
    """
    Uses Conv1d to subsample data followed by Linear and Dropout before Transformer/Convolution Blocks
    (Architecture from Conformer paper: https://arxiv.org/pdf/2005.08100)

    Args:
        input_channels (int): Input of shape (T, N, input_channels).
        output_channels (int): Output of shape (T, N, output_channels)
        dropout (float): default = 0.1
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.ConvSample = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.Linear = nn.Linear(out_channels, out_channels)
        self.Dropout = nn.Dropout(dropout)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Input: (T, N, C_in)
        x = inputs.movedim(0, -1) # (N, C_in, T)
        x = self.ConvSample(x) # (N, C_out, T_out)
        x = x.movedim(-1, 0) # (T_out, N, C_out)
        x = self.Linear(x)
        return self.Dropout(x) 

class CTFeedForward(nn.Module):

    """
    Feedforward Module (LayerNorm -> Linear x 4 -> Swish -> Dropout -> Linear x 1/4 -> Dropout -> + Residuals)
    (Architecture from Conformer paper: https://arxiv.org/pdf/2005.08100)

    Args:
        num_features (int): Input of shape (T, N, num_features) -> (T, N, C = width*channels)
        dropout (float): default = 0.1
    """
    def __init__(
        self,
        num_features: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, 4*num_features), # expand by 4
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4*num_features, num_features), # project to original
            nn.Dropout(dropout),
        )
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Input: (T, N, C)
        x = inputs
        x = self.model(x)
        
        # residuals (skip connection)
        x = inputs + x
        return x

class CTAttention(nn.Module):
    """
    (Architecture from Conformer paper: https://arxiv.org/pdf/2005.08100)

    Args:
        num_features (int): Input of shape (T, N, num_features) -> (T, N, C = width*channels)
        dropout (float): default = 0.1
        max_len (int): default = 5000
        num_heads (int): default = 4, used for MultiHeadAttention

    """
    def __init__(
        self,
        num_features: int,
        dropout: float = 0.1,
        max_len: int = 1024,
        num_heads: int = 4,
        use_mask: bool = True,
    ) -> None:
    
        super().__init__()
        self.LayerNorm = nn.LayerNorm(num_features)
        self.RPA = RelativePositionAttention(d_model=num_features, num_heads=num_heads, max_len=max_len, dropout=dropout, use_mask=use_mask)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        #Input: (T, N, C)
        x = inputs.movedim(0, 1) # (N, T, C)
        x = self.LayerNorm(x)
        x = self.RPA(x) 

        # (T, N, C)
        x = x.movedim(1, 0) 

        # residuals (skip connection)
        x = inputs + x
        return x

class CTConv(nn.Module):
    """
    Convolution Module (Layernorm -> Linear x 2 -> GLU -> Depthwise Conv -> Batchnorm -> Swish -> Linear -> Dropout -> +Residual)
    (Architecture from Conformer paper: https://arxiv.org/pdf/2005.08100)
    
    Uses original TDSConv implementation w/ LayerNorm and ReLU

    Args:
        num_features (int)
        kernel_width (int): The kernel size of the temporal convolution.
        dropout (float): default = 0.1
    """
    def __init__(self, num_features: int, kernel_width: int, dropout: float=0.1) -> None:
        super().__init__()
        self.LayerNorm = nn.LayerNorm(num_features)
        self.LinExpand = nn.Linear(num_features, 2*num_features)
        self.GLU = nn.GLU()
        self.DepthConv = nn.Conv1d(in_channels=num_features, out_channels=num_features, kernel_size=kernel_width, groups=num_features, padding='same')
        self.BatchNorm = nn.BatchNorm1d(num_features)
        self.Swish = nn.SiLU()
        self.Linear = nn.Linear(num_features, num_features)
        self.Dropout = nn.Dropout(dropout)

        #self.Residual = nn.MaxPool1d(kernel_size=kernel_width, stride=1)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Input: (T, N, C)
        x = inputs
        x = self.LayerNorm(x)
        x = self.LinExpand(x)
        x = self.GLU(x)

        x = x.movedim(0, -1) # (N, C, T)
        x = self.DepthConv(x)
        x = self.BatchNorm(x)
        x = self.Swish(x)

        x = x.movedim(-1, 0) # (T, N, C)
        x = self.Linear(x)
        x = self.Dropout(x)

        # residuals (skip connection)
        #res = inputs.movedim(0, -1) # (N, C, T)
        #res = self.Residual(res)
        #res = res.movedim(-1, 0) # (T, N, C)

        #T_out = x.shape[0]
        #x = x + inputs[-T_out:]


        #x = x + res
        x = x + inputs
        return x

class CTBlock(nn.Module):
    """
    Architecture from Conformer paper: https://arxiv.org/pdf/2005.08100
    
    Args:
        num_features (int):
        max_len (int)
        num_heads (int)
        kernel_width (int): The kernel size of the temporal convolution.
        dropout (float): default = 0.1
        use_mask (bool): True
    """
    def __init__(self, 
            num_features: int, 
            kernel_width: int,
            max_len: int = 1024, 
            num_heads: int = 4, 
            dropout: float = 0.1,
            use_mask: bool = True,
    ) -> None:
        super().__init__()
        self.FF1 = CTFeedForward(num_features=num_features, dropout=dropout)
        self.RPA = CTAttention(num_features=num_features, dropout=dropout, max_len=max_len, num_heads=num_heads, use_mask=use_mask)
        self.Conv = CTConv(num_features=num_features, kernel_width=kernel_width, dropout=dropout)
        #self.Res =  nn.MaxPool1d(kernel_size=kernel_width, stride=1)
        self.FF2 = CTFeedForward(num_features=num_features, dropout=dropout)
        self.LayerNorm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Input: (T, N, C)
        x = inputs
        x = x + 0.5 * self.FF1(x)
        x = x + self.RPA(x)

        #res = x.movedim(0, -1) # (N, C, T)
        #res = self.Res(res)
        #res = res.movedim(-1, 0) # (T, N, C)

        
        temp = x
        x = self.Conv(x)
        #x = x + res
        #T_out = x.shape[0]
        #temp = temp[-T_out:]
        x = x + temp

        x = x + 0.5 * self.FF2(x)
        x = self.LayerNorm(x)
        return x

class CTEncoder(nn.Module):
    """
    Architecture from Conformer paper: https://arxiv.org/pdf/2005.08100
    
    Args:
        num_features (int): Input of shape (T, N, num_features) -> (T, N, C = width*channels) = number of features after subsampling
        max_len (int): default = 1024
        num_heads (int): default = 4, used for MultiHeadAttention

        num_layers (int): number of times block is repeated
        kernel_width (int): The kernel size of the temporal convolution.
        dropout (float): default = 0.1
    """
    def __init__(self,
        num_features: int, 
        max_len: int=1024, 
        num_heads: int=4,
        num_layers: int=2, 
        kernel_width: int=32, 
        dropout: float=0.1,
        use_mask: bool=True,
        ) -> None:
        
        super().__init__()        
        tc_blocks: list[nn.Module] = []
        for _ in torch.arange(0, num_layers):
            tc_blocks.extend(
                [
                    CTBlock(num_features=num_features, max_len=max_len, num_heads=num_heads, 
                    kernel_width=kernel_width, dropout=dropout, use_mask=use_mask),
                ]
            )
            
        self.tc_blocks = nn.Sequential(*tc_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tc_blocks(inputs)  # (T, N, num_features)

class AttentionTDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        num_heads: int = 4,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    ConformerMHSA(num_features=num_features, num_heads=num_heads, dropout=dropout),
                    TDSConv2dBlock_v2(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)

class SubsampleMHSA(nn.Module):
    def __init__(
        self,
        num_features: int,
        subsample_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ConvSub = nn.Conv1d(in_channels=num_features, out_channels=subsample_features, kernel_size=1)
        self.MHSA = ConformerMHSA(num_features=subsample_features, num_heads=num_heads, dropout=0.1)
        self.ConvExpand = nn.Conv1d(in_channels=subsample_features, out_channels=num_features, kernel_size=1)
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        x = x.movedim(0, -1)
        x = self.ConvSub(x)
        x = x.movedim(-1, 0)

        x = self.MHSA(x)

        x = x.movedim(0, -1)
        x = self.ConvExpand(x)
        x = x.movedim(-1, 0)

        x = inputs + x
        
        return x  # (T, N, num_features)

class SubAttTDSConvEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        subsample_features: int,
        num_heads: int = 4,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    SubsampleMHSA(num_features=num_features, subsample_features=subsample_features, num_heads=num_heads, dropout=0.1),
                    TDSConv2dBlock_v2(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)

class LSTMEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_features: int,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.LSTM = nn.LSTM(
            input_size=num_features, 
            hidden_size=hidden_features, 
            num_layers=num_layers,
            batch_first=False, 
            bidirectional=True,
            )

        self.fc = TDSFullyConnectedBlock(hidden_features*2)
        self.out = nn.Linear(hidden_features*2, num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x, _ = self.LSTM(inputs)
        x = self.fc(x)
        x = self.out(x)
        return x


