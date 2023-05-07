import argparse

import numpy as np
import scipy.sparse as sp
import torch
from einops import rearrange
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components


class StandardScaler:
    """
        input/output shape [batch_size, channel, n_time, n_vertex]
    """
    def __init__(self, fit_data: torch.Tensor) -> None:
        if len(fit_data.shape) != 3:
            # shape of fit_data [num_of_data, num_vertex, channel]
            fit_data = rearrange(fit_data, 'd t v c -> (d t) v c')

        fit_data = rearrange(fit_data, 't v c -> (t v) c')
        self.mean = torch.mean(fit_data, dim=0)
        self.std = torch.std(fit_data, dim=0)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            # shape of fit_data [num_of_data, num_vertex, channel]
            v = x.shape[1]
            x = rearrange(x, 't v c -> (t v) c')
            x = (x-self.mean)/self.std
            return rearrange(x, '(t v) c -> t v c', v=v).float()
        else:
            # for metro data [day, n_time, n_vertex, channel]
            v = x.shape[-2]
            batch_size = x.shape[0]
            x = rearrange(x, 'd t v c -> (d t v) c')
            x = (x - self.mean) / self.std
            return rearrange(x, '(d t v) c -> d t v c', d=batch_size, v=v).float()

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            # dataset data arrange by [time, vertex, channel]
            v = x.shape[1]
            x = rearrange(x, 't v c -> (t v) c')
            x = x*self.std + self.mean
            return rearrange(x, '(t v) c -> t v c', v=v)
        else:
            # network output data/target data arrange by [batch_size, channel, time, vertex]
            v = x.shape[-1]
            batch_size = x.shape[0]
            x = rearrange(x, 'b c t v -> (b t v) c')
            x = x * self.std + self.mean
            return rearrange(x, '(b t v) c -> b c t v', b=batch_size, v=v)

    def data_info(self) -> (torch.Tensor, torch.Tensor):
        # size of mean is [channel]
        # size of std is [channel]
        return self.mean, self.std


def calc_spatial_emb(adj: np.ndarray, k: int) -> torch.Tensor:
    n_components = connected_components(sp.csr_matrix(adj), directed=False, return_labels=False)
    assert n_components == 1  # the graph should be connected, nonzero value should be 1

    n_vertex = adj.shape[0]
    degree_matrix = np.sum(adj, axis=1) ** (-1 / 2)

    laplacian_matrix = np.eye(n_vertex) - (adj * degree_matrix).T * degree_matrix  # normalized Laplacian

    _, v = eigh(laplacian_matrix)
    spatial_emb = v[:, 1:(k + 1)]  # eigenvectors corresponding to the k smallest non-trivial eigenvalues

    spatial_emb = spatial_emb.astype(dtype=np.float32)
    spatial_emb = torch.from_numpy(spatial_emb).transpose(-1, -2)

    # shape of spatial_emb [k, n_vertex]
    return spatial_emb


def calc_mask(args: argparse.Namespace, device: str):
    n_time = args.n_his

    # 完全的因果遮罩
    temporal_mask = torch.ones(n_time, n_time).float().to(device)
    temporal_mask = torch.triu(temporal_mask, diagonal=1 + n_time // 2)
    temporal_mask = -10e5 * temporal_mask

    # shape of spatial_emb [n_time, n_time]
    return temporal_mask


class MAELoss(torch.nn.Module):
    """
        size of x/input is [batch_size, channel, n_time, n_vertex]
        size of y/output/target [batch_size, channel, n_time, n_vertex]
        Calc MAE by channel
    """
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        # size of mean is [channel]
        # size of std is [channel]
        self.mean = mean
        self.std = std

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x, y = torch.einsum('bctv->btvc', x), torch.einsum('bctv->btvc', y)
        x = x * self.std + self.mean
        y = y * self.std + self.mean
        mae = torch.absolute(x-y)
        return torch.mean(mae)


def calc_metric(y, y_pred, zscore):
    """
        size of y/y_pred [batch_size, channel, n_pred, n_vertex]
    """
    y, y_pred = y.detach().cpu(), y_pred.detach().cpu()

    y = zscore.inverse_transform(y)
    y_pred = zscore.inverse_transform(y_pred)

    metric = Metrics(y, y_pred)
    # metric = EvaluationMetrics(y, y_pred)
    return metric.all()


# Mask target value 0 out
class Metrics(object):
    """
        masked version error functions partly base on PVCGN
        https://github.com/HCPLab-SYSU/PVCGN

        size of output is [batch_size, channel, n_time, n_vertex]
        size of target [batch_size, channel, n_pred, n_vertex]
    """
    def __init__(self, target, output):
        self.target = target
        self.output = output

        # zero value might be slightly change due to Z-score norm
        self.mask = target < 10e-5

    def mse(self):
        mse = torch.square(self.target - self.output)
        mse[self.mask] = 0
        return torch.mean(mse)

    def rmse(self):
        return torch.sqrt(self.mse())

    def mae(self):
        mae = torch.absolute(self.target - self.output)
        mae[self.mask] = 0
        return torch.mean(mae)

    def mape(self):
        mape = torch.absolute((self.target - self.output)/self.target)
        mape[self.mask] = 0
        return torch.mean(mape * 100)

    def all(self):
        rmse = self.rmse().item()
        mae = self.mae().item()
        mape = self.mape().item()

        return rmse, mae, mape
