import torch
import time
import numpy as np
from src.util.util import proj_root_dir
from src.model.ShelterOutcomeModel import ShelterOutcomeModel
from src.args_and_config.args import args
from torch.utils.data import DataLoader
import torch.nn as nn
from src.data_loader import TrainDataset, ValidateDataset, TestDataset


def evaluate(model, features, labels):
    model.eval()

    with torch.no_grad():
        loss_func = nn.L1Loss()
        predicts = model(features)
        loss = loss_func(predicts.to(torch.float32), labels.to(torch.float32))

        return loss.item()


def save_model_parameters(model):
    torch.save(model.state_dict(), proj_root_dir + 'checkpoints/model_parameters.pth')
    print("save model to %s " % (proj_root_dir + 'checkpoints/model_parameters.pth'))


def train():

    cuda = True
    if args.gpu < 0:
        cuda = False


    model = ShelterOutcomeModel()
    if cuda:
        model.cuda()

    for param in model.parameters():
        print(param)
        nn.init.normal_(param, mean=0, std=0.01)


    train_dataset = TrainDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=10)

    validate_dataset = ValidateDataset()
    validate_dataloader = DataLoader(validate_dataset, batch_size=10,)


    loss_func = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    for epoch_i in range(0, args.n_epochs):
        print("Epoch {:05d} training...".format(epoch_i))
        durations = []

        for features_batch, labels_batch in train_dataloader:
            model.train()
            t0 = time.time()

            features_train = features_batch.float()
            labels_train = labels_batch.float()

            if cuda:
                features_train = features_train.to(args.gpu)
                labels_train = labels_train.to(args.gpu)

            # forward
            predicts = model(features_train)
            loss = loss_func(predicts.to(torch.float32), labels_train.to(torch.float32))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            durations.append(time.time() - t0)

        losses = []
        for features_batch_validate, labels_batch_validate in validate_dataloader:

            features_validate = features_batch_validate.float()
            labels_validate = labels_batch_validate.float()

            if cuda:
                features_validate = features_validate.to(args.gpu)
                labels_validate = labels_validate.to(args.gpu)

            loss = evaluate(model, features_validate, labels_validate)
            losses.append(loss)

        print("Epoch {:05d} | Time(s) {:.4f}s | Loss {:.4f} |".format(epoch_i, np.mean(durations), np.mean(losses)))


    save_model_parameters(model)


def main():
    train()


if __name__ == "__main__":
    main()