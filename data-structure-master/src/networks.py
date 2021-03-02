import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math as m


COMPUTATION_DEVICE_NAME = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def get_init(size):
    result = 1. / np.sqrt(size)
    return -result, result


class BaseNet(nn.Module):
    def __init__(self, dimension_size):
        super(BaseNet, self).__init__()

        # input
        self.l1 = nn.Linear(dimension_size, 300)
        # init uniform
        self.l1.weight.data.uniform_(*get_init(self.l1.in_features))
        self.ln1 = nn.LayerNorm(300)

        # l2
        self.l2 = nn.Linear(300, 400)
        self.l2.weight.data.uniform_(*get_init(self.l2.in_features))
        self.ln2 = nn.LayerNorm(400)

        # out
        self.output = nn.Linear(400, 1)
        self.output.weight.data.uniform_(-0.03, 0.03)

        # opt
        self.device = torch.device(COMPUTATION_DEVICE_NAME)

    def forward(self, x):
        x = F.relu(self.ln1(self.l1(x)))
        x = F.relu(self.ln2(self.l2(x)))
        return self.output(x)


class AutoEncoder(nn.Module):
    def __init__(self, input_size, middle_layer):
        super(AutoEncoder, self).__init__()
        self.layer_accelerator = input_size + m.ceil(input_size / 3)

        self.encoder_input = nn.Linear(input_size, self.layer_accelerator)
        # self.encoder_input.weight.data.uniform_(*get_init(self.encoder_input.in_features))
        self.encoder2 = nn.Linear(self.layer_accelerator, self.layer_accelerator)
        # self.encoder2.weight.data.uniform_(*get_init(self.encoder2.in_features))
        self.encoder_output = nn.Linear(self.layer_accelerator, middle_layer)
        # self.encoder_output.weight.data.uniform_(*get_init(self.encoder_output.in_features))

        self.middle = nn.Linear(middle_layer, middle_layer)
        # self.middle.weight.data.uniform_(*get_init(self.middle.in_features))

        self.decoder_input = nn.Linear(middle_layer, self.layer_accelerator)
        # self.decoder_input.weight.data.uniform_(*get_init(self.decoder_input.in_features))
        self.decoder2 = nn.Linear(self.layer_accelerator, self.layer_accelerator)
        # self.decoder2.weight.data.uniform_(*get_init(self.decoder2.in_features))
        self.decoder_output = nn.Linear(self.layer_accelerator, input_size)
        # self.decoder_output.weight.data.uniform_(*get_init(self.decoder_output.in_features))

        self.device = torch.device(COMPUTATION_DEVICE_NAME)

    def forward(self, x):
        x = F.relu(self.encoder_input(x))
        x = F.relu(self.encoder2(x))
        x = F.relu(self.encoder_output(x))
        x = F.relu(self.middle(x))
        x = F.relu(self.decoder_input(x))
        x = F.relu(self.decoder2(x))
        x = F.relu(self.decoder_output(x))
        return x


class LinearNet(nn.Module):
    def __init__(self, input_size, output_size, middle_layer=None):
        super(LinearNet, self).__init__()

        self.input_size = input_size
        self.layer_accelerator = middle_layer if middle_layer else 100 * input_size
        self.output_size = output_size

        self.linear_input = nn.Linear(self.input_size, self.layer_accelerator)

        self.linear2 = nn.Linear(self.layer_accelerator, self.layer_accelerator)
        self.linear3 = nn.Linear(self.layer_accelerator, self.layer_accelerator)
        self.linear4 = nn.Linear(self.layer_accelerator, self.layer_accelerator)
        self.linear5 = nn.Linear(self.layer_accelerator, self.layer_accelerator)
        self.linear6 = nn.Linear(self.layer_accelerator, self.layer_accelerator)
        self.linear7 = nn.Linear(self.layer_accelerator, self.layer_accelerator)
        self.linear8 = nn.Linear(self.layer_accelerator, self.layer_accelerator)
        self.linear9 = nn.Linear(self.layer_accelerator, self.layer_accelerator)

        self.linear_output = nn.Linear(self.layer_accelerator, self.output_size)

        self.device = torch.device(COMPUTATION_DEVICE_NAME)
        self.to(self.device)

    def forward(self, x):
        x = torch.sigmoid(self.linear_input(x))
        x = torch.sigmoid(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        x = torch.sigmoid(self.linear4(x))
        x = torch.sigmoid(self.linear5(x))
        x = torch.sigmoid(self.linear_output(x))
        return x
