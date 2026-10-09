from __future__ import annotations

from typing import Union

from asleep.models import CNNLSTM, weight_init
import torch

dependencies = ["torch"]


def sleepnet(
    pretrained: bool = True,
    my_device: Union[str, torch.device] = "cpu",
    num_classes: int = 2,
    lstm_nn_size: int = 128,
    dropout_p: float = 0.5,
    bi_lstm: bool = True,
    lstm_layer: int = 1,
    local_weight_path: str = "",
) -> CNNLSTM:
    model = CNNLSTM(
        num_classes=num_classes,
        model_device=my_device,
        lstm_nn_size=lstm_nn_size,
        dropout_p=dropout_p,
        bidrectional=bi_lstm,
        lstm_layer=lstm_layer,
    )
    weight_init(model)

    if pretrained:
        if len(local_weight_path) > 0:
            print("Loading local weight from %s" % local_weight_path)
            state_dict = torch.load(local_weight_path,
                                    map_location=torch.device(my_device))
            model.load_state_dict(
                state_dict)
        else:
            checkpoint = 'https://github.com/OxWearables/asleep/' \
                         'releases/download/0.4.9/sleepnet_apr_16_2024.mdl'
            model.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    checkpoint,
                    progress=True,
                    map_location=torch.device(my_device)))
    model.to(torch.device(my_device), dtype=torch.float)
    return model
