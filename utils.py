import os
import random

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from torch.utils.data import DataLoader

import params
from reading_data import My_Dataset


def init_random_seed(manual_seed):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)


def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)


def init_model(net, restore):
    """Init models with cuda and weights."""
    # init weights of model
    net.apply(init_weights)

    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()

    return net


def get_data_loader(path, train_set, train_batch_size, shuffle=True):
    """Get data loader by name."""
    if train_set == True:

        full_data = My_Dataset(path, train_set)
        train_size = int(0.8 * len(full_data))
        test_size = len(full_data) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(full_data, [train_size, test_size])
        # train_loader = DataLoader(dataset=full_data, batch_size=train_batch_size, shuffle=shuffle, num_workers=0)
        train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=shuffle,drop_last=True, num_workers=0)
        test_loader = DataLoader(dataset=test_dataset, batch_size=train_batch_size, shuffle=shuffle,drop_last=True, num_workers=0)
        return train_loader, test_loader
    else:
        train_data = My_Dataset(path, train_set)
        train_loader = DataLoader(dataset=train_data, batch_size=train_batch_size, shuffle=shuffle, num_workers=0)
        return train_loader


def save_model(net, filename):
    """Save trained model."""
    if not os.path.exists(params.model_root):
        os.makedirs(params.model_root)
    torch.save(net.state_dict(),
               os.path.join(params.model_root, filename))
    print("save pretrained model to: {}".format(os.path.join(params.model_root,
                                                             filename)))