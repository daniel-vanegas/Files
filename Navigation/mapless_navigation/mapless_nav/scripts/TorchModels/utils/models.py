#!/usr/bin/env python3

import numpy as np
import torch


class RNNCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity

        if nonlinearity not in ["tanh", "relu"]:
            raise ValueError("Invalid nonlinearity selected for RNN.")

        self.x2h = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.h2h = torch.nn.Linear(hidden_size, hidden_size, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for parameter in self.parameters():
            parameter.data.uniform_(-std, std)

    def forward(self, input, hx=None):
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size)

        hy = self.x2h(input) + self.h2h(hx)

        if self.nonlinearity == "tanh":
            hy = torch.tanh(hy)
        else:
            hy = torch.relu(hy)

        return hy


class RNN(torch.nn.Module):
    def __init__(
        self,
        channels=5,
        img_size=80,
        hidden_size=500,
        num_layers=3,
        bias=True,
        output_size=3,
        activation="tanh",
    ):
        super().__init__()

        self.name = "RNN"
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size
        self.img_size = img_size
        self.conv_channels = 4

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=self.conv_channels, kernel_size=3)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=7, stride=3)

        conv_size = img_size - 3 + 1
        pooled_size = ((conv_size - 7) // 3) + 1
        conv_output_size = self.conv_channels * pooled_size * pooled_size
        target_size = 2 * img_size

        self.input_size = conv_output_size + target_size

        print(f"conv_channels: {self.conv_channels}")
        print(f"conv_output_size: {conv_output_size}")
        print(f"target_size: {target_size}")
        print(f"RNN input_size: {self.input_size}")

        self.rnn_cell_list = torch.nn.ModuleList()

        if activation == "tanh":
            nonlinear_function = "tanh"
        elif activation == "relu":
            nonlinear_function = "relu"
        else:
            raise ValueError("Invalid activation.")

        self.rnn_cell_list.append(
            RNNCell(self.input_size, self.hidden_size, self.bias, nonlinear_function)
        )

        for _ in range(1, self.num_layers):
            self.rnn_cell_list.append(
                RNNCell(self.hidden_size, self.hidden_size, self.bias, nonlinear_function)
            )

        self.fc = torch.nn.Linear(self.hidden_size, self.output_size)
        self.flat1 = torch.nn.Linear(self.hidden_size, 120)
        self.flat2 = torch.nn.Linear(120, 80)
        self.out = torch.nn.Linear(80, 3)

        self.dropout_20 = torch.nn.Dropout(p=0.2)
        self.dropout_15 = torch.nn.Dropout(p=0.15)
        self.dropout_10 = torch.nn.Dropout(p=0.1)

    def forward(self, x, hx=None):
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [batch, C, 82, 80]. Got {tuple(x.shape)}")

        batch_size = x.size(0)
        sequence_length = x.size(1)

        if x.size(2) != 82:
            raise ValueError(f"Expected height 82, got {x.size(2)}")

        if x.size(3) != 80:
            raise ValueError(f"Expected width 80, got {x.size(3)}")

        target = x[:, :, -2:, :]
        target = torch.flatten(target, start_dim=2)

        image = x[:, :, :-2, :]
        image = image.reshape(batch_size * sequence_length, 1, self.img_size, self.img_size)
        image = self.conv1(image)
        image = self.max_pool(image)
        image = torch.relu(image)
        image = torch.flatten(image, start_dim=1)
        image = image.reshape(batch_size, sequence_length, -1)

        input_sequence = torch.cat([image, target], dim=2)

        if input_sequence.size(2) != self.input_size:
            raise RuntimeError(
                f"Unexpected RNN input size. Expected {self.input_size}, got {input_sequence.size(2)}."
            )

        if hx is None:
            hidden = [
                input_sequence.new_zeros(batch_size, self.hidden_size)
                for _ in range(self.num_layers)
            ]
        else:
            hidden = [hx[layer] for layer in range(self.num_layers)]

        for t in range(sequence_length):
            for layer in range(self.num_layers):
                if layer == 0:
                    hidden[layer] = self.rnn_cell_list[layer](
                        input_sequence[:, t, :], hidden[layer]
                    )
                else:
                    hidden[layer] = self.rnn_cell_list[layer](
                        hidden[layer - 1], hidden[layer]
                    )

        output = hidden[-1]
        output = self.flat1(output)
        output = torch.relu(output)
        output = self.dropout_20(output)
        output = self.flat2(output)
        output = torch.relu(output)
        output = self.dropout_15(output)
        output = self.out(output)
        output = torch.tanh(output)

        return output