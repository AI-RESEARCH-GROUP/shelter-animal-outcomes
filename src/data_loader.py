import numpy as np
from src.util.util import proj_root_dir
from torch.utils.data.dataset import Dataset


def load_total_data():
    all_data = np.load(proj_root_dir + 'data/all_data.npz')
    Y_data = np.load(proj_root_dir + 'data/Y.npz')

    ani_list = all_data["ani_list"]
    sex_list = all_data["sex_list"]
    age_list = all_data["age_list"]
    bre_list = all_data["bre_list"]
    col_list = all_data["col_list"]


    return total_data


total_data = load_total_data()
n_train = 26729
n_test = 11456


class MyDataset(Dataset):
    def __init__(self, ):
        super(MyDataset, self).__init__()
        self.start_index = 0

    def __getitem__(self, index):
        features = total_data[self.start_index + index]['features']
        labels = total_data[self.start_index + index]['labels']

        return features, labels



class TrainDataset(MyDataset):
    def __init__(self):
        super(TrainDataset, self).__init__()
        self.start_index = 0

    def __len__(self):
        return n_train


class TestDataset(MyDataset):
    def __init__(self, ):
        super(TestDataset, self).__init__()
        self.start_index = n_train

    def __len__(self):
        return n_test


def main():
    load_total_data()


if __name__ == "__main__":
    main()