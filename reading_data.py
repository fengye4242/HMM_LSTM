import hdf5storage
import numpy as np
import torch
from torch.utils.data import Dataset
import params
# mat= hdf5storage.loadmat('data/subject_1.mat')
# numat = np.array(mat['train_data'])
# tensor_data=torch.from_numpy(numat)

class My_Dataset(Dataset):
    def __init__(self , data_path, train = True):
        super(My_Dataset, self).__init__()
        data = hdf5storage.loadmat(data_path)
        train_data = data['train_data']
        train_label = data['train_label']-1
        train_mode = data['train_mode']-1
        train_subject = data['train_subject']-1
        train_trial = data['train_trial']-1

        test_data = data['test_data']
        test_label = data['test_label']-1
        test_mode = data['test_mode']-1
        test_subject = data['test_subject']-1
        test_trial = data['test_trial']-1


        if train:


            self.data = torch.from_numpy(train_data).float()
            self.label = torch.from_numpy(train_label).long()
            self.mode = torch.from_numpy(train_mode).float()
            self.sub = torch.from_numpy(train_subject).float()
            self.trial = torch.from_numpy(train_trial).float()

        elif not train:

            self.data = torch.from_numpy(test_data).float()
            self.label = torch.from_numpy(test_label).long()
            self.mode = torch.from_numpy(test_mode).float()
            self.sub = torch.from_numpy(test_subject).float()
            self.trial = torch.from_numpy(test_trial).float()

    def __getitem__(self, item):
        img = self.data[item]
        label = self.label[item]
        mode = self.mode[item]
        sub_label = self.sub[item]
        trial=self.trial[item]
        return img, label, mode, sub_label, trial

    def __len__(self):
        return len(self.data)