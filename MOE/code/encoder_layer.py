#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder self-attention layer definition."""

import torch
from torch import nn
import torch.nn.functional as F

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention as MH


class EncoderLayer(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        stochastic_depth_rate (float): Proability to skip this layer.
            During training, the layer may skip residual computation and return input
            as-is with given probability.
        use_codebooks (bool): Whether to use codebook based cross attention for this layer or not.
            if True, cross attention block is initialized and used. 
    """

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        feed_forward_macaron,
        conv_module,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
        use_codebooks=False,
        no_accents=5
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayer, self).__init__()
        
        self.use_codebooks = use_codebooks
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = LayerNorm(size)  # for the FNN module
        self.norm_mha = LayerNorm(size)  # for the MHA module
        if self.use_codebooks:
            self.norm_codebook = LayerNorm(size)
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = LayerNorm(size)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = LayerNorm(size)  # for the CNN module
            self.norm_final = LayerNorm(size)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate

        if self.use_codebooks:
            self.no_accents = no_accents
            # self.codebook_attentions = nn.ModuleList([MH(1, size, dropout_rate) for _ in range(no_accents)])
            self.codebook_attention = MH(1, size, dropout_rate)

        self.gating_linear = torch.nn.Sequential(
            torch.nn.Linear(size, 2048),
            torch.nn.GELU(),
            torch.nn.Linear(2048, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, 5)
        )

        # self.conv_gating_layer = nn.Conv1d(in_channels=size, out_channels=size, kernel_size=16, stride=16)

    def forward(self, x_input, mask, codebooks, probs, y_probs, cache=None):
        """Compute encoded features.

        Args:
            x_input (Union[Tuple, torch.Tensor]): Input tensor w/ or w/o pos emb.
                - w/ pos emb: Tuple of tensors [(#batch, time, size), (1, time, size)].
                - w/o pos emb: Tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, 1, time).
            codebooks (torch.Tensor): Codebook tensor for the input (#batch, codebooks_per_accent, size).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, 1, time).
            torch.Tensor: Codebooks tensor (#batch, codebooks_per_accent, size).

        """
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if ((skip_layer) and (probs is not None)):
            probs = y_probs
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            if pos_emb is not None:
                return (x, pos_emb), mask, codebooks, probs, 0.0
            return x, mask, codebooks, probs, 0.0
        # if ((skip_layer) and (probs is not None)):
        #     if cache is not None:
        #         x = torch.cat([cache, x], dim=1)
        #     if pos_emb is not None:
        #         return (x, pos_emb), mask, codebooks, probs, 0.0
        #     return x, mask, codebooks, probs, 0.0


        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(
                self.feed_forward_macaron(x)
            )
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if pos_emb is not None:
            x_att = self.self_attn(x_q, x, x, pos_emb, mask)
        else:
            x_att = self.self_attn(x_q, x, x, mask)
            
        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            x = residual + stoch_layer_coeff * self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # ## Changes start
        # if(gating_layer is not None):
        # probs = F.softmax(self.gating_linear(torch.mean(self.conv_gating_layer(x.permute(0, 2, 1)).permute(0,2,1), dim=1)), dim=-1)
        probs = F.softmax(self.gating_linear(torch.mean(x, dim=1)), dim=-1)
            # TODO: SANITY CHECK OF WHAT THE DISTRIBUTION IN PROBS LOOKS LIKE FOR A FEW SAMPLE UTTERANCES
            # codebooks = torch.sum(probs.view(probs.size(0),-1,1,1) * codebooks, dim=1)
        # else:
        #     probs = None
        # ## Changes end
        
        if self.use_codebooks:
            if self.normalize_before:
                x = self.norm_codebook(x)
                
            residual = x

            # l1_loss = 0.0
            # codebook_context = torch.zeros_like(x)
            # for i in range(self.no_accents):
            #     # print("Hey", self.codebook_attention(x, codebooks[:,i,:,:], codebooks[:,i,:,:], None).shape)
            #     codebook_context += probs[:,i].view(probs.size(0), 1, 1) * self.codebook_attention(x, codebooks[:,i,:,:], codebooks[:,i,:,:], None)
            codebook_context, l1_loss = self.codebook_attention(x, codebooks, codebooks, None, probs)
            # codebook_context = self.codebook_attention(x, codebooks, codebooks, None, y_probs)
            # codebook_context, l1_loss = self.codebook_attention(x, codebooks, codebooks)
            
            x = residual + stoch_layer_coeff * self.dropout(codebook_context)
                
            if not self.normalize_before:
                x = self.norm_codebook(x)

        # convolution module
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x = residual + stoch_layer_coeff * self.dropout(self.conv_module(x))
            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(
            self.feed_forward(x)
        )
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        if pos_emb is not None:
            return (x, pos_emb), mask, codebooks, probs, l1_loss

        return x, mask, codebooks, probs, l1_loss
