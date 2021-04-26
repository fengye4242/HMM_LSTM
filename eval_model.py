import torch.nn as nn
import torch

def eval_model(model,data_loader):

    loss = 0
    acc = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for step, (data, label, mode, sub_label, trial) in enumerate(data_loader):
            input_data=data.permute(2,0,1)
            input_label=label.squeeze_().long()

            tag_scores = model(input_data.cuda())
            loss = criterion(tag_scores, input_label.cuda())

            pred_cls = tag_scores .data.max(1)[1]
            acc += pred_cls.eq(label.data.cuda()).cpu().sum().item()
    loss /= len(data_loader)
    acc /= len(data_loader.dataset)
    print("Avg src Loss = {}, Avg src Accuracy = {:2%}".format(loss, acc))
    return loss,acc




