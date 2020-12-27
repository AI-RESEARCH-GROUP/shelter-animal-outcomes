import time
import torch
import numpy as np
from src.util.util import proj_root_dir, file_exists
from src.model.ShelterOutcomeModel import ShelterOutcomeModel
from src.args_and_config.args import args
from torch.utils.data import DataLoader
import torch.nn as nn
from src.data_loader import TestDataset


def evaluate(model, features, labels):
    model.eval()

    with torch.no_grad():
        loss_func = nn.L1Loss()
        predicts = model(features)
        loss = loss_func(predicts.to(torch.float32), labels.to(torch.float32))

        return loss.item()


def test():
    if not file_exists(proj_root_dir + 'checkpoints/model_parameters.pth'):
        print()
        print("please run train.py first !!!")
        print()
        exit(-1)

    cuda = False
    # if args.gpu < 0:
    #     cuda = False

    model = ShelterOutcomeModel()
    model.load_state_dict(torch.load(proj_root_dir + 'checkpoints/model_parameters.pth'))
    if cuda:
        model.cuda()


    test_dataset = TestDataset()
    test_dataloader = DataLoader(test_dataset, batch_size=1,)

    t0 = time.time()
    durations = []

    losses = []
    for features_batch_test, labels_batch_test in test_dataloader:

        features_test = features_batch_test.float()
        labels_test = labels_batch_test.float()


        if cuda:
            features_test = features_test.to(args.gpu)
            labels_test = labels_test.to(args.gpu)

        loss = evaluate(model, features_test, labels_test)
        losses.append(loss)

    durations.append(time.time() - t0)

    print()
    print("Test loss {:.4f} | Time(s) {:.4f}s".format(np.mean(losses), np.mean(durations) / 1000))


def main():
    test()


if __name__ == '__main__':
    main()