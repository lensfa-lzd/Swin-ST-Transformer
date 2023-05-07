import argparse
import math
import os

import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from model.model import SwinSTTransformer
from script.dataloader import load_adj, load_data, data_transform
from script.utility import calc_spatial_emb, calc_mask, calc_metric, MAELoss, StandardScaler
from script.visualize import progress_bar

cpu_num = 8  # limit cpu
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)


# New feature for RTX GPUs based on pytorch 2.0
# torch.set_float32_matmul_precision('high')


def get_parameters() -> (argparse.Namespace, str):
    parser = argparse.ArgumentParser(description='Swin-ST-Transformer')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--dataset', type=str, default='pems-bay')
    parser.add_argument('--his', type=int, default=60, help='minute')
    parser.add_argument('--pred', type=int, default=60, help='minute')
    parser.add_argument('--time_intvl', type=int, default=5, help='means N minutes')
    parser.add_argument('--n_block', type=int, default=4)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_channel', type=int, default=16)
    parser.add_argument('--window_size', type=int, default=40)
    parser.add_argument('--k', type=int, default=16, help='k smallest non-trivial eigenvalues')
    parser.add_argument('--ste_channel', type=int, default=16)

    parser.add_argument('--lr', type=float, default=10e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--compile', type=bool, default=False, help='New feature on pytorch 2.0')

    args_ = parser.parse_args()
    args_.n_his = int(args_.his / args_.time_intvl)
    args_.n_pred = int(args_.pred / args_.time_intvl)

    args_.in_channel = args_.out_channel = 1

    print('Training configs: {}'.format(args_))

    # Running in Nvidia GPU (CUDA) or CPU
    if args_.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device_ = torch.device('cuda')
    else:
        device_ = torch.device('cpu')

    return args_, device_


def data_prepare() -> (torch.Tensor, StandardScaler, DataLoader, DataLoader, DataLoader):
    global args
    global device

    adj, n_vertex = load_adj(args.dataset)
    args.n_vertex = n_vertex

    # shape of se_ [k, padded_vertex]
    se_ = calc_spatial_emb(adj, args.k).to(device)

    args.mask = calc_mask(args, device)

    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, args.dataset)

    num_of_data = torch.load(os.path.join(dataset_path, 'vel.pth')).shape[0]

    train_ratio = 7
    val_ratio = 1
    test_ratio = 2

    ratio_sum = train_ratio + val_ratio + test_ratio

    train_ratio /= ratio_sum
    val_ratio /= ratio_sum
    test_ratio /= ratio_sum

    len_val = int(math.floor(num_of_data * val_ratio))
    len_test = int(math.floor(num_of_data * test_ratio))
    len_train = int(num_of_data - len_val - len_test)

    train_tuple, val_tuple, test_tuple = load_data(args.dataset, len_train, len_val)

    train_data, val_data, test_data = train_tuple[0], val_tuple[0], test_tuple[0]
    train_te, val_te, test_te = train_tuple[1], val_tuple[1], test_tuple[1]

    zscore_ = StandardScaler(train_data)

    # shape of train/val/test [num_of_data, num_vertex, channel]
    train_data = zscore_.transform(train_data)
    val_data = zscore_.transform(val_data)
    test_data = zscore_.transform(test_data)

    train_data = torch.cat((train_data, train_te), dim=-1).float()
    val_data = torch.cat((val_data, val_te), dim=-1).float()
    test_data = torch.cat((test_data, test_te), dim=-1).float()

    # size of input/x is [batch_size, channel, n_time, n_vertex]
    # size of y/target [batch_size, channel, n_time, n_vertex]
    x_train, y_train = data_transform(train_data, args.n_his, args.n_pred)
    x_val, y_val = data_transform(val_data, args.n_his, args.n_pred)
    x_test, y_test = data_transform(test_data, args.n_his, args.n_pred)

    y_train, y_val, y_test = \
        y_train[:, :args.in_channel, :, :], y_val[:, :args.in_channel, :, :], y_test[:, :args.in_channel, :, :]

    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    train_data = TensorDataset(x_train, y_train)
    val_data = TensorDataset(x_val, y_val)
    test_data = TensorDataset(x_test, y_test)

    train_iter_ = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    val_iter_ = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
    test_iter_ = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    # size of x/input is [batch_size, channel, n_time, n_vertex]
    # size of y/target [batch_size, channel, n_time, n_vertex]
    return se_, zscore_, train_iter_, val_iter_, test_iter_


def prepare_model() -> (torch.nn.Module, torch.nn.Module, torch.nn.Module, str):
    global args
    global device
    global zscore

    mean, std = zscore.data_info()
    mean, std = mean.to(device), std.to(device)
    loss_ = MAELoss(mean, std)

    ckpt_name_ = f'head_{args.n_head}_channel_{args.n_channel}_ckpt.pth'
    model_ = SwinSTTransformer(args).to(device)

    if args.compile:
        print('Using compile feature')
        model_ = torch.compile(model_)

    optimizer_ = optim.Adam(model_.parameters(), lr=args.lr)

    return loss_, model_, optimizer_, ckpt_name_


def train() -> None:
    """
        size of x/input is [batch_size, channel, n_time, n_vertex]
        size of y/output/target [batch_size, channel, n_time, n_vertex]
    """
    global loss_function, args, optimizer, model, train_iter, val_iter, test_iter, se, zscore, ckpt_name

    best_point = 10e5
    model_dist = model.state_dict()

    for epoch in range(args.epochs):
        l_sum = 0.0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()
        for batch_idx, (x, y) in enumerate(train_iter):
            y_pred = model(x, se)
            loss = loss_function(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            l_sum += loss.item()

            rmse, mae, mape = calc_metric(y, y_pred, zscore)

            if batch_idx % 20 == 0:
                progress_bar(batch_idx, len(train_iter), 'Train loss: %.3f | mae, mape, rmse: %.3f, %.1f%%, %.3f'
                             % (l_sum / (batch_idx + 1), mae, mape, rmse))

        print('epoch', epoch + 1)
        val_loss, val_mae = evaluation(model_dist, val_iter, phase='validation')
        print()

        if val_mae < best_point:
            best_point = val_mae
            model_dist = model.state_dict()

    test_loss, test_mae = evaluation(model_dist, test_iter, phase='test')


def evaluation(
        model_dist,
        data_iter: DataLoader,
        phase: str,
        saved: bool = True
) -> (float, float):
    global model, ckpt_name, zscore, args, se, loss_function

    model.eval()
    l_sum, mae_sum = 0.0, 0.0

    if phase == 'test':
        # load best model in train
        model.load_state_dict(model_dist)
        print('Test: best train model loaded')

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data_iter):
            y_pred = model(x, se)
            loss = loss_function(y_pred, y)
            l_sum += loss.item()

            rmse, mae, mape = calc_metric(y, y_pred, zscore)
            mae_sum += mae
            if batch_idx % 20 == 0:
                progress_bar(batch_idx, len(data_iter), str(phase) +
                             ' loss: %.3f | mae, mape, rmse: %.3f, %.1f%%, %.3f'
                             % (l_sum / (batch_idx + 1), mae, mape, rmse))

        print()
        val_mae = mae_sum / len(data_iter)

        if saved and phase == 'test':
            try:
                checkpoint = torch.load('./checkpoint/' + ckpt_name)
                val_loss = checkpoint['loss(mae)']
                print('Found local model ckpt')
                if val_mae < val_loss:
                    print('Get better model, saving')
                    save_model(val_mae)
            except Exception as err:
                save_model(val_mae)
                print(str(err))
                print('Local model ckpt not found, saving...')

        print(str(phase) + '_mae', val_mae)
    return l_sum / len(data_iter), val_mae


def save_model(val_loss: float) -> None:
    global args, model

    ckpt = model.state_dict()
    if args.compile:
        names = []
        for param_name_old in ckpt:
            # print(param_name)
            # 去除torch.compile保存字典中的_orig_mod
            # _orig_mod.TransformerBlock.3.GCTAtt_Block.LayerNorm2.weight
            _, param_name_new = param_name_old.split('.', 1)
            names.append((param_name_old, param_name_new))

        for old_name, new_name in names:
            ckpt[new_name] = ckpt.pop(old_name)

    checkpoint = {
        'config_args': args,
        'net': ckpt,
        'loss(mae)': val_loss,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(checkpoint, './checkpoint/' + ckpt_name)


if __name__ == '__main__':
    args, device = get_parameters()
    se, zscore, train_iter, val_iter, test_iter = data_prepare()
    loss_function, model, optimizer, ckpt_name = prepare_model()

    train()
