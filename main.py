import sys, os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import argparse
from data import CPIDataset
from models.core import *
from utils import *
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score, accuracy_score
import random
import torch.backends.cudnn as cudnn
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# training function at each epoch
def train_dta(model, loss_fn, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    model.train()

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        # print(data.y.view(-1, 1).float().to(device))

        loss = loss_fn(output, data['LABEL'].view(-1, 1).float().cuda())
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data['LABEL']),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def predicting_dta(model, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data['LABEL'].view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


def train_dti(model, loss_fn, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    model.train()
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data['LABEL'].cuda())
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data['LABEL']),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def predicting_dti(model, loader):
    model.eval()
    total_preds = []
    total_labels = []

    print('Making predictions for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            output = model(data)
            total_preds.append(output.cpu().numpy())
            total_labels.append(data['LABEL'].cpu().numpy())

    return np.concatenate(total_labels), np.concatenate(total_preds)


def davis_dataloader(seed=0, batch_size=256, workers=4, dataset = 'davis', data_path='./data'):
    print('\nrunning on ', dataset)

    path = data_path + '/' + dataset

    data = CPIDataset(f'{path}.csv', f'{path}/compound', f'{path}/protein')

    print(len(data))

    test_size = (int)(len(data) * 0.1)

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset=data,
        lengths=[len(data) - (test_size * 2), test_size * 2],
        generator=torch.Generator().manual_seed(seed)
    )
    val_dataset, test_dataset = torch.utils.data.random_split(
        dataset=test_dataset,
        lengths=[test_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=data.collate_fn,
        shuffle=True,
        num_workers=workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=data.collate_fn,
        shuffle=True,
        num_workers=workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=data.collate_fn,
        shuffle=True,
        num_workers=workers,
    )

    return train_loader, val_loader, test_loader

def others_dataloader(batch_size, workers, dataset = 'davis', data_path='./data'):
    print('\nrunning on ', dataset)

    path = data_path + '/' + dataset

    if dataset == 'bindingdb':
        train_set = CPIDataset(f'{path}_train.csv', f'{path}/compound', f'{path}/protein/train')
        test_set = CPIDataset(f'{path}_test.csv',f'{path}/compound', f'{path}/protein/test')

    else:

        train_set = CPIDataset(f'{path}_train.csv', f'{path}/compound', f'{path}/protein')

        test_set = CPIDataset(f'{path}_test.csv',f'{path}/compound', f'{path}/protein')

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        collate_fn=train_set.collate_fn,
        shuffle=True,
        num_workers=workers,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        collate_fn=test_set.collate_fn,
        shuffle=True,
        num_workers=workers,
    )

    return train_loader, test_loader



def run(args: argparse.Namespace):


    data_path = args.root_data_path
    dataset = args.dataset
    batch_size = args.batch_size
    LR = args.learning_rate
    NUM_EPOCHS = args.max_epochs
    seed = args.seed

    model = PMMRNet(args).cuda()


    print('Learning rate: ', LR)
    print('Epochs: ', NUM_EPOCHS)

    # Main program: iterate over different datasets
    if dataset == 'davis':
        train_loader, val_loader, test_loader = davis_dataloader(seed, batch_size, args.num_workers, dataset, data_path)
    else:
        train_loader,test_loader = others_dataloader(batch_size, args.num_workers, dataset, data_path)


    if args.objective == 'regression':
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=30,
                                                               eps=1e-08)
        best_mae = 1000
        best_ci = 0
        best_epoch = -1
        model_file_name = 'model_' + dataset + '.pth'
        result_file_name = 'result_' + dataset + '.csv'
        for epoch in range(NUM_EPOCHS):

            train_dta(model, loss_fn, train_loader, optimizer, epoch + 1)

            if dataset == 'davis':
                G, P = predicting_dta(model, val_loader)
            else:
                G, P = predicting_dta(model, test_loader)

            ret = [rmse(G, P), mae(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]
            # ret = [rmse(G, P), mse(G, P), ci(G, P), rm2(G, P)]

            current_lr = optimizer.param_groups[0]['lr']
            print("current lr:", current_lr)
            with open(result_file_name, 'a') as f:
                f.write(str(epoch) + ',' + ','.join(map(str, ret)) + '\n')
            if ret[1] < best_mae:
                torch.save(model.state_dict(), model_file_name)
                best_epoch = epoch + 1
                best_mae = ret[1]
                best_ci = ret[-1]
                print('mae improved at epoch ', best_epoch, '; best_mae,best_ci:', best_mae, best_ci)
            else:
                print(ret[1], 'No improvement since epoch ', best_epoch, '; best_mae,best_ci:', best_mae, best_ci)

            scheduler.step(best_mae)


    else:
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=30,
                                                               eps=1e-08)

        best_aupr = 0
        best_epoch = -1
        model_file_name = f'{dataset}_{seed}.pth'
        result_file_name = f'result_{dataset}_{seed}.csv'

        for epoch in range(NUM_EPOCHS):
            train_dti(model, loss_fn, train_loader, optimizer, epoch + 1)
            y_true, y_scores = predicting_dti(model, test_loader)

            y_pred = np.argmax(y_scores, axis=1)
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            aupr = auc(recall, precision)

            fpr, tpr, _ = roc_curve(y_true, y_pred)
            auroc = auc(fpr, tpr)

            f1 = f1_score(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)

            ret = [acc, aupr, auroc, f1]
            with open(result_file_name, 'a') as f:
                f.write(f'{epoch},{",".join(map(str, ret))}\n')
            current_lr = optimizer.param_groups[0]['lr']
            print("current lr:", current_lr)
            if aupr > best_aupr:
                torch.save(model.state_dict(), model_file_name)
                best_epoch = epoch + 1
                best_aupr = aupr

                print('AUPR improved at epoch', best_epoch, f'Best AUPR: {best_aupr}')
            else:
                print(aupr, f'No improvement since epoch {best_epoch}. Best AUPR: {best_aupr}')

            scheduler.step(best_aupr)



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_data_path', type=str, default='./data', help='Raw Data Path')
    parser.add_argument('--dataset', type=str, default='pdb', help='Datasets')
    parser.add_argument('--objective',
                        type=str,
                        default='regression',
                        help='Objective (classification / regression)')
    parser.add_argument('--seed', type=int, default=0, help='Random Seed')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch Size for Train(Validation/Test)')
    parser.add_argument('--max_epochs', type=int, default=200, help='Max Trainning Epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of Subprocesses for Data Loading')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning Rate for Trainning')
    parser.add_argument('--decoder_layers', type=int, default=3, help='Number of Layers for Decoder')
    parser.add_argument('--linear_heads', type=int, default=10, help='Number of linear attention heads')
    parser.add_argument('--linear_hidden_dim', type=int, default=32, help='Dimension of linear attention heads')
    parser.add_argument('--decoder_heads', type=int, default=4, help='Number of headers in the decoder')
    parser.add_argument('--encoder_heads', type=int, default=4, help='Number of Transformer heads')
    parser.add_argument('--gnn_layers', type=int, default=3, help='Layers of GNN')
    parser.add_argument('--encoder_layers', type=int, default=1, help='Layers of Transformer')
    parser.add_argument('--decoder_nums', type=int, default=1, help='Layers of Decoder')
    parser.add_argument('--decoder_dim', type=int, default=128, help='Dimension of Decoder')
    parser.add_argument('--compound_gnn_dim', type=int, default=78, help='Hidden Dimension for Attention')
    parser.add_argument('--pf_dim', type=int, default=1024, help='Hidden Dimension for Positional Feed Forward')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout Rate')
    parser.add_argument('--protein_dim', type=int, default=128, help='Dimension for Protein')
    parser.add_argument('--compound_structure_dim', type=int, default=78, help='Dimension for Compound Structure')
    parser.add_argument('--compound_text_dim', type=int, default=128, help='Dimension for Compound Text')
    parser.add_argument('--compound_pretrained_dim', type=int, default=384, help='Dimension of pretrained for '
                                                                                 'compound language model')
    parser.add_argument('--protein_pretrained_dim', type=int, default=480, help='Dimension of pretrained for protein '
                                                                                'language model')




    return parser.parse_args()

if __name__ == '__main__':
    params = parse_args()
    print(params)
    torch.cuda.manual_seed_all(params.seed)
    run(params)

