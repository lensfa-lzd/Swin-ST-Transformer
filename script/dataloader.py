import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from einops import repeat


def load_adj(dataset_name: str) -> (np.ndarray, int):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    adj = torch.load(os.path.join(dataset_path, 'adj.pth'))
    adj = adj.numpy()

    n_vertex = 325
    identity_matrix = np.eye(n_vertex)

    # Remove self-connection
    adj = adj - identity_matrix

    adj[adj > 0] = 1  # 0/1 matrix, symmetric

    # adj is weighted in dataset above
    return adj, n_vertex


def load_data(dataset_name: str, len_train: int, len_val: int) -> (Tuple, Tuple, Tuple):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)

    # shape of vel [num_of_data, num_vertex, channel]
    vel = torch.load(os.path.join(dataset_path, 'vel.pth'))
    n_vertex = vel.shape[-2]

    time_index = pd.read_hdf(os.path.join(dataset_path, 'time_index.h5'))
    time_index = calc_te(time_index)

    # shape of time_index [num_of_data, 2]
    # te [..., 0] for day of week
    # te [..., 1] for time of day
    te = repeat(time_index, 't c -> t v c', v=n_vertex)

    train = (vel[: len_train], te[: len_train])
    val = (vel[len_train: len_train + len_val], te[len_train: len_train + len_val])
    test = (vel[len_train + len_val:], te[len_train + len_val:])
    return train, val, test


def calc_te(time_index: pd.DataFrame) -> torch.Tensor:
    # shape of te day_of_week/time_of_day [num_of_data, 1]
    day_of_week = time_index.dayofweek.values
    time_of_day = time_index.timeofday.values.astype(np.int64)
    day_of_week, time_of_day = torch.from_numpy(day_of_week), torch.from_numpy(time_of_day)
    day_of_week, time_of_day = day_of_week.unsqueeze(dim=-1), time_of_day.unsqueeze(dim=-1)

    return torch.cat((day_of_week, time_of_day), dim=-1)


def data_transform(data: torch.Tensor, n_his: int, n_pred: int) -> (torch.Tensor, torch.Tensor):
    # produce data slices for x_data and y_data

    # shape of data [num_of_data, num_vertex, channel]
    channel = data.shape[-1]
    n_vertex = data.shape[1]
    len_record = data.shape[0]
    num = len_record - n_his - n_pred

    x = torch.zeros([num, n_his, n_vertex, channel])
    y = torch.zeros([num, n_pred, n_vertex, channel])

    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = data[head: tail]
        y[i, :, :, :] = data[tail: tail + n_pred]

    x = torch.einsum('btvc->bctv', x).float()
    y = torch.einsum('btvc->bctv', y).float()

    # size of input/x is [batch_size, channel, n_time, n_vertex]
    # size of y/target [batch_size, channel, n_time, n_vertex]
    return x, y
