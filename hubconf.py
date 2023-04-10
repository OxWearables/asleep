import os
from src.asleep.models import CNNLSTM, weight_init
import torch

dependencies = ["torch"]


def sleepnet(pretrained=True, my_device="cpu", class_num=2, lstm_nn_size=128,
             dropout_p=0.5, bi_lstm=True, lstm_layer=1):
    model = CNNLSTM(
        num_classes=class_num,
        model_device=my_device,
        lstm_nn_size=lstm_nn_size,
        dropout_p=dropout_p,
        bidrectional=bi_lstm,
        lstm_layer=lstm_layer,
    )
    weight_init(model)

    if pretrained:
        checkpoint = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint,
                                                                 progress=True,
                                                                 map_location=torch.device(my_device)))
    model.to(my_device, dtype=torch.float)
    return model
