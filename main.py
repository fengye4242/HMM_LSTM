import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from reading_data import My_Dataset
import params
from torch.utils.data import DataLoader
from utils import init_random_seed , get_data_loader, init_model
from utils import make_variable, save_model
from models import LSTM
import fitlog
from eval_model import eval_model
if __name__ == '__main__':
    fitlog.set_log_dir('logs/', True)
    fit_msg = 'LSTM_train'
    fitlog.commit(__file__, fit_msg)

    init_random_seed(params.manual_seed)

    # train_data=My_Dataset(params.src_dataset)
    train_loader,test_loader = get_data_loader(params.src_dataset,train_set=True,train_batch_size=params.batch_size,shuffle=True)
    # train_loader = DataLoader(dataset=train_data, batch_size=params.batch_size, shuffle=True, num_workers=0)
    # test_loader = DataLoader(dataset=train_data, batch_size=params.batch_size, shuffle=True, num_workers=0)

    model = LSTM(input_dim=6, hidden_dim=64,output_size=4,layers=2)
    model.cuda()
    # model.float()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    fitlog.add_hyper(params.num_epochs, name='train_epoch')
    for epoch in range(params.num_epochs):
        for step,(data, label, mode, sub_label, trial) in enumerate(train_loader):
            input_data=data.permute(2,0,1).cuda()
            input_label=label.squeeze_().long().cuda()
            model.zero_grad()
            model.hidden = model.init_hidden()

            # model.cuda()
            tag_scores = model(input_data)

            loss = loss_function(tag_scores, input_label)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = eval_model(model,train_loader)
        test_loss,  test_acc= eval_model(model,test_loader)
        fitlog.add_loss(train_loss, step=epoch, name='train_loss')
        fitlog.add_metric(train_loss, step=epoch, name='train_acc')
        fitlog.add_loss(test_loss, step=epoch, name='test_loss')
        fitlog.add_metric(test_acc, step=epoch, name='test_acc')

        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(model, "LSTM-{}.pt".format(epoch + 1))
    fitlog.finish()
    model_save='LSTM-final.pt'








