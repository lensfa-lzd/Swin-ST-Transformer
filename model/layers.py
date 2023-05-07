import math

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn


class FFN(nn.Module):
    """
    Pipeline
        In
            Fully-Connected Layer
            relu activation function
            Fully-Connected Layer
        Out
    """

    def __init__(self, n_channel: int) -> None:
        super(FFN, self).__init__()
        self.fc1 = Linear(in_features=n_channel, out_features=4 * n_channel)
        self.act = nn.ReLU()
        self.fc2 = Linear(in_features=4 * n_channel, out_features=n_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x


class GlobalTemporalAtt(nn.Module):
    """
        Rearrange x from [batch_size, channel, n_time, n_vertex]
        to [batch_size, n_vertex, n_time, channel]
        and conduct attention alone temporal dimension

        return size [batch_size, channel, n_time, n_vertex]
    """

    def __init__(self, n_channel: int, n_head: int, mask: torch.Tensor):
        super(GlobalTemporalAtt, self).__init__()
        att_channel = int(n_channel / n_head)
        self.att_channel = att_channel
        self.mask = mask

        self.to_qkv = nn.Linear(in_features=att_channel, out_features=3 * att_channel)
        self.split_head = Rearrange('b (h c) t v -> (b h) v t c', h=n_head)
        self.concat_head = Rearrange('(b h) v t c -> b (h c) t v', h=n_head)
        self.proj = Linear(in_features=n_channel, out_features=n_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [b, c, t, v]
        x = self.split_head(x)
        # size of q/k/v is [batch_size*head, n_vertex, n_time, channel/head]
        qkv = self.to_qkv(x)
        q = qkv[..., :self.att_channel]
        k = qkv[..., self.att_channel:2 * self.att_channel]
        v = qkv[..., -self.att_channel:]

        att_dimension = q.shape[-1]
        att_map = torch.matmul(q, k.transpose(-1, -2))
        att_map /= (att_dimension ** 0.5)

        # Temporal Causal Mask
        att_map += self.mask

        att_map = F.softmax(att_map, dim=-1)
        x = torch.matmul(att_map, v)

        x = self.concat_head(x)
        x = self.proj(x)

        return x


class WindowAtt(nn.Module):
    """
        Rearrange x from [batch_size, channel, n_time, n_vertex]
        to [batch_size, Window_size, n_time*n_vertex/Window_size, channel]
        and conduct attention within last two dimension

        return size [batch_size, channel, n_time, n_vertex]
    """

    def __init__(
            self,
            n_channel: int,
            n_head: int,
            window_size: int,
    ):
        # size of input/x is [batch_size, channel, n_time, n_vertex]
        super(WindowAtt, self).__init__()
        att_channel = int(n_channel / n_head)
        self.att_channel = att_channel
        self.window_size = window_size

        self.to_qkv = nn.Linear(in_features=att_channel, out_features=3 * att_channel)
        self.split_head = Rearrange('b (h c) t v -> (b h) c t v', h=n_head)
        self.concat_head = Rearrange('(b h) c t v -> b (h c) t v', h=n_head)
        self.proj = Linear(in_features=n_channel, out_features=n_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_size = 0
        # x [b, c, t, v] -> [b*h, c/h, t, v]
        x = self.split_head(x)
        n_vertex = x.shape[-1]

        # Padding
        if n_vertex % self.window_size != 0:
            pad_size = self.window_size * math.ceil(n_vertex / self.window_size) - n_vertex
            x = self.shift_pad(x, pad_size)

        # x [b*h, c/h, t, v] -> [b*h, v/w, t*w, c/h]
        #   b c t v -> b v/w t*w c
        x = self.split_windows(x, self.window_size)

        # size of q/k/v is
        # [batch_size*head, n_vertex/Window_size(n_window), n_time*window_size, channel/head]
        qkv = self.to_qkv(x)
        q = qkv[..., :self.att_channel]
        k = qkv[..., self.att_channel:2 * self.att_channel]
        v = qkv[..., -self.att_channel:]

        att_dimension = q.shape[-1]
        att_map = torch.matmul(q, k.transpose(-1, -2))
        att_map /= (att_dimension ** 0.5)

        att_map = F.softmax(att_map, dim=-1)
        x = torch.matmul(att_map, v)

        x = self.concat_windows(x, self.window_size)
        if pad_size != 0:
            x = self.del_pad(x, pad_size)

        x = self.concat_head(x)
        x = self.proj(x)

        return x

    @staticmethod
    def split_windows(x: torch.Tensor, window_size: int) -> torch.Tensor:
        """
            v/w: number of windows
            w: window size
            b c t v -> b v/w t*w c
        """
        x = x.unsqueeze(dim=0)
        x_window = torch.cat(torch.split(x, window_size, dim=-1), dim=0)
        x_window = rearrange(x_window, 'w b c t v -> b w (t v) c')

        return x_window

    @staticmethod
    def concat_windows(x: torch.Tensor, window_size: int) -> torch.Tensor:
        """
            v/w: number of windows
            w: window size
            b v/w t*w c -> b c t v
        """
        x = x.unsqueeze(dim=0)
        x = torch.cat(torch.split(x, window_size, dim=-2), dim=0)  # t b v/w w c
        x_concat = x[:, :, 0, :, :]
        for i in range(1, x.shape[2]):
            x_concat = torch.cat((x_concat, x[:, :, i, :, :]), dim=-2)
        x_concat = rearrange(x_concat, 't b v c -> b c t v')

        return x_concat

    @staticmethod
    def shift_pad(x: torch.Tensor, pad_size: int) -> torch.Tensor:
        """
            x [batch_size, channel, n_time, n_vertex]
            padding in a cycling manner
        """

        return torch.cat((x, x[..., :pad_size]), dim=-1)

    @staticmethod
    def del_pad(x: torch.Tensor, pad_size: int) -> torch.Tensor:
        return x[..., :-pad_size]


class OutputProj(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, in_time: int, out_time: int) -> None:
        super(OutputProj, self).__init__()
        self.reshape_1 = Rearrange('b c t v -> b v (c t)')
        self.proj_1 = nn.Linear(in_features=in_time * in_channel, out_features=out_time * out_channel)
        self.reshape_2 = Rearrange('b v (c t) -> b c t v', c=out_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reshape_1(x)
        x = self.proj_1(x)
        x = self.reshape_2(x)

        return x


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum('bctv->bvtc', x)
        x = self.linear(x)
        x = torch.einsum('bvtc->bctv', x)

        return x


def shift_window(x: torch.Tensor, shift_pixel: int) -> torch.Tensor:
    """
    Shift attention window by
    shifting the data/image itself in a cycling manner

    x  [batch_size, channel, n_time, n_vertex]
    """
    x = torch.cat((x, x[..., :shift_pixel]), dim=-1)
    x = x[..., shift_pixel:]

    return x
