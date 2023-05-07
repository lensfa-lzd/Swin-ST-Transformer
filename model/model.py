import argparse
from argparse import Namespace

import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn

from model.layers import WindowAtt, FFN, shift_window, OutputProj, Linear, GlobalTemporalAtt


# from layers import WindowAtt, FFN, shift_window, OutputProj, Linear, GlobalTemporalAtt


class SwinSTTransformerModule(nn.Module):
    """
    Overall pipeline
        In
            STS-Att
            Layer Normalization
            Feed Forward Neural Network
            Layer Normalization
            Shift Window

            STS-Att
            Layer Normalization
            Feed Forward Neural Network
            Layer Normalization
            Shift Window
        Out
    """

    def __init__(self, args: argparse.Namespace) -> None:
        # size of input/x is [batch_size, channel, n_time, n_vertex]
        super(SwinSTTransformerModule, self).__init__()
        n_channel = args.n_channel
        n_time = args.n_his
        n_vertex = args.n_vertex
        n_head = args.n_head
        window_size = args.window_size
        shift_pixel = int(window_size / 2)
        norm_shape = [n_channel, n_time, n_vertex]

        self.shift_pixel = shift_pixel
        self.sts_att_a = WindowAtt(n_channel, n_head, window_size)
        self.layer_norm_a1 = nn.LayerNorm(norm_shape)
        self.ffn_a = FFN(n_channel)
        self.layer_norm_a2 = nn.LayerNorm(norm_shape)

        self.sts_att_b = WindowAtt(n_channel, n_head, window_size)
        self.layer_norm_b1 = nn.LayerNorm(norm_shape)
        self.ffn_b = FFN(n_channel)
        self.layer_norm_b2 = nn.LayerNorm(norm_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # size of input/x is [batch_size, channel, n_time, n_vertex]
        x = self.sts_att_a(x) + x
        x = self.layer_norm_a1(x)
        x = self.ffn_a(x) + x
        x = self.layer_norm_a2(x)
        x = shift_window(x, self.shift_pixel)

        x = self.sts_att_b(x) + x
        x = self.layer_norm_b1(x)
        x = self.ffn_b(x) + x
        x = self.layer_norm_b2(x)
        x = shift_window(x, self.shift_pixel)

        return x


class GlobalTemporalCasualModule(nn.Module):
    """
    Overall pipeline
        GCT-Att
        Layer Normalization
        Feed Forward Neural Network
        Layer Normalization

        norm_shape = (n_channel, args.n_his, n_vertex)
    """

    def __init__(self, args: argparse.Namespace) -> None:
        # size of input/x is [batch_size, channel, n_time, n_vertex]
        super(GlobalTemporalCasualModule, self).__init__()
        n_channel = args.n_channel
        n_time = args.n_his
        n_vertex = args.n_vertex
        n_head = args.n_head
        temporal_mask = args.mask
        # use_temporal_mask = args.use_temporal_mask

        norm_shape = [n_channel, n_time, n_vertex]

        self.gtc_att = GlobalTemporalAtt(n_channel, n_head, temporal_mask)
        self.layer_norm1 = nn.LayerNorm(norm_shape)
        self.ffn = FFN(n_channel)
        self.layer_norm2 = nn.LayerNorm(norm_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # size of input/x is [batch_size, channel, n_time, n_vertex]
        x = self.gtc_att(x) + x
        x = self.layer_norm1(x)
        x = self.ffn(x) + x
        x = self.layer_norm2(x)

        return x


class TransformerBlock(nn.Module):
    """
    Overall pipeline
        In
            swin_st_transformer_module + \
                global_temporal_casual_module
        Out
    """

    def __init__(self, args: argparse.Namespace) -> None:
        # size of input/x is [batch_size, channel, n_time, n_vertex]
        super(TransformerBlock, self).__init__()
        self.swin_st_transformer_module = SwinSTTransformerModule(args)
        self.global_temporal_casual_module = GlobalTemporalCasualModule(args)

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:

        x_t = self.global_temporal_casual_module(x)
        x = self.swin_st_transformer_module(x)
        return x + x_t


class SwinSTTransformer(nn.Module):
    """
    Overall pipeline
        In
            input_layer
            + st_position_embedding(Spatial-Temporal Embedding)
            transformer_block
            dimension_adapter
        Out
    """

    def __init__(self, args: Namespace) -> None:
        super(SwinSTTransformer, self).__init__()
        in_channel = args.in_channel
        n_channel = args.n_channel
        n_time = args.n_his
        n_block = args.n_block
        out_channel = args.out_channel
        out_time = args.n_pred

        self.in_channel = args.in_channel
        self.input_layer = Linear(in_features=in_channel, out_features=n_channel)

        self.st_position_embedding = \
            STE(n_time=n_time, in_channel=n_channel, n_time_of_day=int(60 * 24 / args.time_intvl))

        self.transformer_block = nn.ModuleList([TransformerBlock(args) for _ in range(n_block)])

        self.dimension_adapter = \
            OutputProj(in_channel=n_channel, out_channel=out_channel, in_time=n_time, out_time=out_time)

    def forward(
            self,
            x_in: torch.Tensor,
            se: torch.Tensor,
    ) -> torch.Tensor:
        """
        x_in (x_data, te)
        size of input/x is [batch_size, channel, n_time, n_vertex]
        size of y/target [batch_size, channel, n_time, n_vertex]
        size of ste is [n_time, n_vertex]
        """
        x, te = x_in[:, :self.in_channel, :, :], x_in[:, -2:, :, :]
        ste = self.st_position_embedding(te, se)

        x = self.input_layer(x)

        x = x + ste

        for block in self.transformer_block:
            x = block(x)

        x = self.dimension_adapter(x)
        return x


# TODO 对模块进行重新命名，使之与论文中的一致
# TODO 完善注释
class STE(nn.Module):
    def __init__(
            self,
            n_time: int,
            in_channel: int,
            n_time_of_day: int,
            se_channel: int = 16,  # value by default
            te_channel: int = 16  # value by default
    ) -> None:
        super(STE, self).__init__()
        self.n_time = n_time
        self.n_time_of_day = n_time_of_day
        self.proj_te = Linear(in_features=n_time_of_day + 7, out_features=te_channel)
        self.proj_ste = Linear(in_features=te_channel + se_channel, out_features=in_channel)

    def forward(self, te_raw: torch.Tensor, se: torch.Tensor) -> torch.Tensor:
        """
        split data and temporal embedding

            shape of te_raw [batch_size, n_time, n_vertex, channel]
            # channel 0 for dayofweek
            # channel 1 for timeofday

            size of se [channel, n_vertex]
        """
        te_raw = te_raw.long()
        # shape of day_of_week/time_of_day [batch_size, n_time, n_vertex, channel]
        day_of_week = F.one_hot(te_raw[:, 0, :, :], num_classes=7)
        time_of_day = F.one_hot(te_raw[:, 1, :, :], num_classes=self.n_time_of_day)
        te = torch.cat((day_of_week, time_of_day), dim=-1).float()
        te = torch.einsum('btvc->bctv', te)
        te = self.proj_te(te)
        se = repeat(se, 'c v -> b c t v', t=self.n_time, b=te.shape[0])

        # print(te.shape, se.shape)
        # assert se.shape[1] == te.shape[1] == 16
        ste = torch.cat((te, se), dim=1)
        ste = self.proj_ste(ste)
        return ste
