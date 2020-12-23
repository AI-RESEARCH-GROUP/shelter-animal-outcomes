import torch
import time
import numpy as np
from src.util.util import proj_root_dir
from src.model import ShelterOutcomeModel
from src.args_and_config.args import args
from torch.utils.data import Dataset, DataLoader
import torch.optim as torch_optim
import torch.nn as nn
import torch.nn.functional as F
from src.data_loader import load_total_data, TrainDataset, TestDataset
from torchvision import models
from datetime import datetime


def save_model_parameters(model):
    torch.save(model.state_dict(), proj_root_dir + 'checkpoints/model_parameters.pth')
    print("save model to %s " % (proj_root_dir + 'checkpoints/model_parameters.pth'))


def train():
    cuda = True
    if args.gpu < 0:
        cuda = False

    total_data = load_total_data()
    features_0 = total_data[0]['features']
    labels_0 = total_data[0]['labels']
    in_feats = features_0.shape[1]
    out_feats = labels_0.shape[1]

    model = ShelterOutcomeModel(in_feats,
                    args.n_hidden,
                    out_feats,
                    args.n_layers,
                    F.relu,)

    if cuda:
        model.cuda()

    # ================================================
    # 4) model parameter init
    # ================================================
    for param in model.parameters():
        print(param)
        nn.init.normal_(param, mean=0, std=0.01)


    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ================================================
    # 5) train loop
    # ================================================

    train_dataset = TrainDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=1)

    test_dataset = TestDataset()
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    for epoch_i in range(0, args.n_epochs):
        print("Epoch {:05d} training...".format(epoch_i))
        durations = []

        for features_batch, labels_batch in train_dataloader:
            model.train()
            t0 = time.time()

            # =========================
            # get input parameter
            # =========================
            # because batch_size = 1, so just pick the first element
            features = features_batch[0].int()
            labels = labels_batch[0].int()

            if cuda:
                features = features.to(args.gpu)
                labels = labels.to(args.gpu)


            # forward
            predicts = model(features)
            loss = loss_func(predicts.to(torch.float32), labels.to(torch.float32))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            durations.append(time.time() - t0)


    losses = []
    for g_adj_batch_test, features_batch_test, labels_batch_test in test_dataloader:
        # because batch_size = 1, so just pick the first element
        g_adj_test = g_adj_batch_test[0].int()
        features_test = features_batch_test[0].int()
        labels_test = labels_batch_test[0].int()

        if cuda:
            features_test = features_test.to(args.gpu)
            labels_test = labels_test.to(args.gpu)

        losses.append(loss)

    print()
    print("Test loss {:.4f}".format(np.mean(losses)))

    # ================================================
    # 8) save model parameters
    # ================================================
    save_model_parameters(model)


def main():
    train()


if __name__ == "__main__":
    main()


def main():
    train()


if __name__ == "__main__":
    main()