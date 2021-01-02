import numpy as np
from src.util.util import proj_root_dir
from torch.utils.data.dataset import Dataset
from src.args_and_config.config import config
import math


def train_validate_split(train_validate_data):
    train_validate_ratio = config["train_validate_ratio"].split(":")

    train_ratio = float(train_validate_ratio[0]) / (float(train_validate_ratio[0])
                                                         + float(train_validate_ratio[1]))
    validate_ratio = float(train_validate_ratio[1]) / (float(train_validate_ratio[0])
                                                            + float(train_validate_ratio[1]))
    n_train = math.floor(len(train_validate_data) * train_ratio)
    n_train = n_train if n_train > 0 else 1
    n_validate = math.floor(len(train_validate_data) * validate_ratio)
    n_validate = n_validate if n_validate > 0 else 1

    return n_train, n_validate


def load_total_data():

    train_validate_data = np.load(proj_root_dir + 'data/all_data.npz')['train_df']
    test_data = np.load(proj_root_dir + 'data/all_data.npz')['test_df']
    return train_validate_data, test_data


train_validate_data, test_data = load_total_data()
n_train, n_validate = train_validate_split(train_validate_data)
n_test = len(test_data)


class TrainDataset(Dataset):
    def __init__(self):
        super(TrainDataset, self).__init__()
        self.start_index = 0

    def __len__(self):
        return n_train

    def __getitem__(self, index):
        features = train_validate_data[self.start_index + index][1:]
        labels = train_validate_data[self.start_index + index][0]
        return features, labels

class ValidateDataset(Dataset):
    def __init__(self, ):
        super(ValidateDataset, self).__init__()
        self.start_index = n_train

    def __len__(self):
        return n_validate

    def __getitem__(self, index):
        features = train_validate_data[self.start_index + index][1:]
        labels = train_validate_data[self.start_index + index][0]
        return features, labels


class TestDataset(Dataset):
    def __init__(self, ):
        super(TestDataset, self).__init__()
        self.start_index = 0

    def __len__(self):
        return n_test

    def __getitem__(self, index):
        features = test_data[self.start_index + index]
        return features


def main():
    load_total_data()


if __name__ == "__main__":
    main()