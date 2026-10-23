#!/usr/bin/env python3
"""Mapless RNN model definitions.

One module supports two checkpoint layouts:
- ``ros1_legacy``: historical ROS 1 architecture, state-dict compatible with the
  original 2-channel/4-layer/350-hidden-unit network.
- ``ros2``: the newer per-frame convolution architecture used by existing ROS 2
  checkpoints.

New training defaults to ``ros1_legacy`` for both data formats; this makes the
historical architecture the standard while allowing the modern variant to be
selected explicitly for experiments or loading old checkpoints.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class RNNCell(nn.Module):
    """Original vanilla RNN cell; names kept stable for checkpoint compatibility."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
    ) -> None:
        super().__init__()
        if nonlinearity not in ("tanh", "relu"):
            raise ValueError("nonlinearity must be 'tanh' or 'relu'")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.bias = bool(bias)
        self.nonlinearity = nonlinearity
        self.x2h = nn.Linear(self.input_size, self.hidden_size, bias=self.bias)
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -std, std)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor | None = None) -> torch.Tensor:
        if hidden is None:
            hidden = inputs.new_zeros(inputs.size(0), self.hidden_size)
        result = self.x2h(inputs) + self.h2h(hidden)
        return torch.tanh(result) if self.nonlinearity == "tanh" else torch.relu(result)


class RNN(nn.Module):
    """Configurable mapless RNN; ``architecture`` chooses the conv feature layout.

    Args are deliberately compatible with the historical constructor. Defaults
    match the ROS 1 model: C=2, hidden_size=350, num_layers=4.
    """

    ARCHITECTURES = ("ros1_legacy", "ros2")

    def __init__(
        self,
        channels: int = 2,
        img_size: int = 80,
        hidden_size: int = 350,
        num_layers: int = 4,
        bias: bool = True,
        output_size: int = 3,
        activation: str = "tanh",
        architecture: str = "ros1_legacy",
        dropout_1: float = 0.20,
        dropout_2: float = 0.15,
    ) -> None:
        super().__init__()
        if architecture not in self.ARCHITECTURES:
            raise ValueError(f"architecture must be one of {self.ARCHITECTURES}; got {architecture!r}")
        if channels < 1 or img_size < 1 or hidden_size < 1 or num_layers < 1:
            raise ValueError("channels, img_size, hidden_size and num_layers must be positive")
        if not (0.0 <= dropout_1 < 1.0 and 0.0 <= dropout_2 < 1.0):
            raise ValueError("dropout rates must be in [0, 1)")
        if activation not in ("tanh", "relu"):
            raise ValueError("activation must be 'tanh' or 'relu'")

        self.name = "RNN"
        self.architecture = architecture
        self.channels = int(channels)
        self.img_size = int(img_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bias = bool(bias)
        self.output_size = int(output_size)
        self.activation = activation
        self.dropout_1_rate = float(dropout_1)
        self.dropout_2_rate = float(dropout_2)
        self.max_pool = nn.MaxPool2d(kernel_size=7, stride=3)

        # 80x80 -> 78x78 after conv(k=3) -> 24x24 after max-pool(7,3).
        conv_size = self.img_size - 2
        pooled_size = ((conv_size - 7) // 3) + 1
        if pooled_size <= 0:
            raise ValueError(f"img_size={img_size} is too small for conv/pooling layers")
        pooled_features = pooled_size * pooled_size
        target_features = 2 * self.img_size

        if architecture == "ros1_legacy":
            # Original ROS 1 semantics: the stacked C maps become Conv2d input
            # channels and the C learned filters form the recurrent time axis.
            # This preserves old state-dict tensor names and shapes.
            self.conv1 = nn.Conv2d(self.channels, self.channels, kernel_size=3)
            recurrent_input_size = pooled_features + target_features
            self.out_ch = self.channels
        else:
            # Existing ROS 2 checkpoint format: each timestep is convolved as
            # a single image with four learned feature maps.
            self.conv_channels = 4
            self.conv1 = nn.Conv2d(1, self.conv_channels, kernel_size=3)
            recurrent_input_size = self.conv_channels * pooled_features + target_features
            self.out_ch = self.conv_channels

        self.input_size = int(recurrent_input_size)
        self.rnn_cell_list = nn.ModuleList()
        cell_activation = activation
        self.rnn_cell_list.append(
            RNNCell(self.input_size, self.hidden_size, self.bias, cell_activation)
        )
        for _ in range(1, self.num_layers):
            self.rnn_cell_list.append(
                RNNCell(self.hidden_size, self.hidden_size, self.bias, cell_activation)
            )

        # Retained even though the original forward does not use fc: older
        # checkpoints contain these parameters and strict loading requires them.
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.flat1 = nn.Linear(self.hidden_size, 120)
        self.flat2 = nn.Linear(120, 80)
        self.out = nn.Linear(80, self.output_size)
        self.dropout_20 = nn.Dropout(p=self.dropout_1_rate)
        self.dropout_15 = nn.Dropout(p=self.dropout_2_rate)
        self.dropout_10 = nn.Dropout(p=0.1)

    def _legacy_features(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = x.shape[:2]
        if sequence_length != self.channels:
            raise ValueError(
                f"ROS 1 legacy architecture binds sequence_length to channels: "
                f"model channels={self.channels}, input sequence={sequence_length}."
            )
        target = torch.flatten(x[:, :, -2:, :], start_dim=2)
        image = x[:, :, :-2, :]
        image = self.conv1(image)
        image = self.max_pool(image)
        image = torch.relu(image)
        image = torch.flatten(image, start_dim=2)
        features = torch.cat((image, target), dim=2)
        if features.shape[-1] != self.input_size:
            raise RuntimeError(f"Legacy model expected feature size {self.input_size}, got {features.shape[-1]}")
        return features

    def _ros2_features(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = x.shape[:2]
        target = torch.flatten(x[:, :, -2:, :], start_dim=2)
        image = x[:, :, :-2, :].reshape(
            batch_size * sequence_length, 1, self.img_size, self.img_size
        )
        image = self.conv1(image)
        image = self.max_pool(image)
        image = torch.relu(image)
        image = torch.flatten(image, start_dim=1)
        image = image.reshape(batch_size, sequence_length, -1)
        features = torch.cat((image, target), dim=2)
        if features.shape[-1] != self.input_size:
            raise RuntimeError(f"ROS 2 model expected feature size {self.input_size}, got {features.shape[-1]}")
        return features

    def forward(self, x: torch.Tensor, hx=None) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input [batch, C, 82, 80], got {tuple(x.shape)}")
        if x.shape[2:] != (self.img_size + 2, self.img_size):
            raise ValueError(
                f"Expected input spatial shape {(self.img_size + 2, self.img_size)}, got {tuple(x.shape[2:])}"
            )

        features = self._legacy_features(x) if self.architecture == "ros1_legacy" else self._ros2_features(x)
        batch_size, sequence_length = features.shape[:2]
        if hx is None:
            hidden = [features.new_zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        else:
            if isinstance(hx, torch.Tensor):
                hidden = [hx[layer] for layer in range(self.num_layers)]
            else:
                hidden = list(hx)
            if len(hidden) != self.num_layers:
                raise ValueError(f"Expected {self.num_layers} hidden states, got {len(hidden)}")

        for step in range(sequence_length):
            for layer in range(self.num_layers):
                layer_input = features[:, step, :] if layer == 0 else hidden[layer - 1]
                hidden[layer] = self.rnn_cell_list[layer](layer_input, hidden[layer])

        output = torch.relu(self.flat1(hidden[-1]))
        output = self.dropout_20(output)
        output = torch.relu(self.flat2(output))
        output = self.dropout_15(output)
        return torch.tanh(self.out(output))
