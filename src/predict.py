import time
import torch
import numpy as np
from src.util.util import proj_root_dir, file_exists
from src.model.ShelterOutcomeModel import ShelterOutcomeModel
from src.args_and_config.args import args
from torch.utils.data import DataLoader
from src.data_loader import TestDataset


def predict():
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

    for features_batch_test, labels_batch_test in test_dataloader:

        features_test = features_batch_test.float()

        if cuda:
            features_test = features_test.to(args.gpu)

    predicts = model(features_test)
    durations.append(time.time() - t0)

    print(predicts)
    print("Time(s) {:.4f}s".format(np.mean(durations) / 1000))


def main():
    predict()


if __name__ == '__main__':
    main()