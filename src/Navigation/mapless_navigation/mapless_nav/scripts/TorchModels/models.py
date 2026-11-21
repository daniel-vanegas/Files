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
        img_h: int = 80,
        img_w: int = 80,
        hidden_size: int = 350,
        num_layers: int = 4,
        bias: bool = True,
        output_size: int = 3,
        activation: str = "tanh",
        architecture: str = "ros1_legacy",
        dropout_1: float = 0.20,
        dropout_2: float = 0.15,
        img_size: int | None = None,
    ) -> None:
        super().__init__()
        if img_size is not None:
            # Legacy keyword retained for old checkpoints.
            img_h = img_size
            img_w = img_size
        if architecture not in self.ARCHITECTURES:
            raise ValueError(f"architecture must be one of {self.ARCHITECTURES}; got {architecture!r}")
        if channels < 1 or img_h < 1 or img_w < 1 or hidden_size < 1 or num_layers < 1:
            raise ValueError("channels, img_h, img_w, hidden_size and num_layers must be positive")
        if not (0.0 <= dropout_1 < 1.0 and 0.0 <= dropout_2 < 1.0):
            raise ValueError("dropout rates must be in [0, 1)")
        if activation not in ("tanh", "relu"):
            raise ValueError("activation must be 'tanh' or 'relu'")

        self.name = "RNN"
        self.architecture = architecture
        self.channels = int(channels)
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bias = bool(bias)
        self.output_size = int(output_size)
        self.activation = activation
        self.dropout_1_rate = float(dropout_1)
        self.dropout_2_rate = float(dropout_2)
        self.max_pool = nn.MaxPool2d(kernel_size=7, stride=3)

        # Conv(k=3) then max-pool(7,3) applied independently on each axis.
        conv_h = self.img_h - 2
        conv_w = self.img_w - 2
        pooled_h = ((conv_h - 7) // 3) + 1
        pooled_w = ((conv_w - 7) // 3) + 1
        if pooled_h <= 0 or pooled_w <= 0:
            raise ValueError(f"img_h={img_h}, img_w={img_w} is too small for conv/pooling layers")
        pooled_features = pooled_h * pooled_w
        target_features = 2 * self.img_w

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
            batch_size * sequence_length, 1, self.img_h, self.img_w
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

    def forward(self, x: torch.Tensor, hx=None, return_state: bool = False):
        if x.ndim != 4:
            raise ValueError(f"Expected input [batch, C, {self.img_h + 2}, {self.img_w}], got {tuple(x.shape)}")
        if x.shape[2:] != (self.img_h + 2, self.img_w):
            raise ValueError(
                f"Expected input spatial shape {(self.img_h + 2, self.img_w)}, got {tuple(x.shape[2:])}"
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
        out_final = torch.tanh(self.out(output))
        
        if return_state:
            return out_final, hidden
        return out_final


class TemporalPolicy(nn.Module):
    """Frame-wise CNN followed by a recurrent policy; no global map or external memory.

    Input: [batch, steps, grid_height + 2, grid_width]. The two trailing
    rows hold the repeated relative target (distance and angle). They are
    read from column zero, without altering the recorded T-LOM.
    """

    MODEL_TYPES = ("rnn", "gru", "lstm")

    def __init__(self, model_type: str = "rnn", img_h: int = 100,
                 img_w: int = 80, hidden_size: int = 350,
                 num_layers: int = 4, dropout_1: float = 0.20,
                 dropout_2: float = 0.15):
        super().__init__()
        if model_type not in self.MODEL_TYPES:
            raise ValueError(f"Unknown temporal model: {model_type}")
        if min(img_h, img_w, hidden_size, num_layers) < 1:
            raise ValueError("Dimensions and recurrent depth must be positive")
        self.model_type = model_type
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        recurrent = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[model_type]
        self.recurrent = recurrent(
            input_size=16 * 4 * 4 + 2, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout_1 if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 120), nn.ReLU(),
            nn.Dropout(dropout_1), nn.Linear(120, 80), nn.ReLU(),
            nn.Dropout(dropout_2), nn.Linear(80, 3), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, hx=None, return_state: bool = False):
        if x.ndim != 4 or tuple(x.shape[2:]) != (self.img_h + 2, self.img_w):
            raise ValueError(
                f"Expected [batch,steps,{self.img_h+2},{self.img_w}], got {tuple(x.shape)}"
            )
        batch, steps = x.shape[:2]
        if steps < 1:
            raise ValueError("At least one observation is required")
        image = x[:, :, :self.img_h, :].reshape(batch * steps, 1, self.img_h, self.img_w)
        features = self.encoder(image).flatten(1).reshape(batch, steps, -1)
        target = x[:, :, -2:, 0]
        temporal_input = torch.cat((features, target), dim=-1)
        sequence, next_state = self.recurrent(temporal_input, hx)
        prediction = self.head(sequence[:, -1, :])
        if return_state:
            return prediction, next_state
        return prediction


class ConvLSTMCell(nn.Module):
    """Spatial LSTM cell. Hidden and cell state stay [B, channels, h, w]."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("ConvLSTM kernel_size must be odd")
        self.hidden_channels = int(hidden_channels)
        self.gates = nn.Conv2d(input_channels + hidden_channels, 4 * hidden_channels,
                               kernel_size=kernel_size, padding=kernel_size // 2)
        # Encourage keeping information at the beginning of training.
        with torch.no_grad():
            self.gates.bias[self.hidden_channels:2 * self.hidden_channels].fill_(1.0)

    def forward(self, x: torch.Tensor, state=None):
        if state is None:
            shape = (x.shape[0], self.hidden_channels, x.shape[-2], x.shape[-1])
            h, c = x.new_zeros(shape), x.new_zeros(shape)
        else:
            h, c = state
        i, f, o, g = self.gates(torch.cat((x, h), dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class GoalConditionedConvLSTM(nn.Module):
    """Fuse target spatially *before* recurrent processing.

    Input is the existing [B, T, H + 2, W] frame format. The last two
    rows hold distance and theta. No flatten or global pooling takes
    place before ConvLSTM. The returned state can be reused across ROS
    callbacks, or detached between training chunks of the same episode.
    """

    def __init__(self, img_h: int = 100, img_w: int = 80,
                 spatial_hidden_channels: int = 32, num_layers: int = 1,
                 dropout_1: float = 0.20, dropout_2: float = 0.15,
                 encoder_channels: tuple[int, int] = (8, 16),
                 goal_fusion: str = "dual"):
        super().__init__()
        if min(img_h, img_w, spatial_hidden_channels, num_layers) < 1:
            raise ValueError("Dimensions and recurrent depth must be positive")
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.spatial_hidden_channels = int(spatial_hidden_channels)
        self.num_layers = int(num_layers)
        if len(encoder_channels) != 2 or any(int(c) < 1 for c in encoder_channels):
            raise ValueError("encoder_channels must contain two positive filter counts")
        self.encoder_channels = tuple(int(c) for c in encoder_channels)
        if goal_fusion not in ("dual", "mixed"):
            raise ValueError("goal_fusion must be dual or mixed")
        self.goal_fusion = goal_fusion
        intermediate, last = self.encoder_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(1, intermediate, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(intermediate, last, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )
        self.spatial_cells = nn.ModuleList([
            ConvLSTMCell(last + (2 if self.goal_fusion == "dual" else 1) if idx == 0 else spatial_hidden_channels,
                         spatial_hidden_channels)
            for idx in range(num_layers)
        ])
        # Spatial compression is allowed only AFTER recurrent processing.
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(spatial_hidden_channels * 4 * 4, 120), nn.ReLU(),
            nn.Dropout(dropout_1), nn.Linear(120, 80), nn.ReLU(),
            nn.Dropout(dropout_2), nn.Linear(80, 3), nn.Tanh(),
        )

    @staticmethod
    def make_target_maps(target: torch.Tensor, height: int, width: int,
                         mode: str = "dual") -> torch.Tensor:
        """Encode (distance, theta) into spatial maps without pooling/flattening.

        In mixed mode the upper ceil(H/2) rows contain distance, while
        lower rows contain theta.  For 4x4 and (2, .5), the result is
        [[2,2,2,2], [2,2,2,2], [.5,.5,.5,.5], [.5,.5,.5,.5]].
        """
        if target.shape[-1] != 2 or height < 2 or width < 1:
            raise ValueError("Expected two target values and an image at least two rows tall")
        if mode == "dual":
            return target[..., None, None].expand(*target.shape[:-1], 2, height, width)
        if mode == "mixed":
            top = (height + 1) // 2
            d = target[..., 0, None, None, None].expand(*target.shape[:-1], 1, top, width)
            theta = target[..., 1, None, None, None].expand(*target.shape[:-1], 1, height-top, width)
            return torch.cat((d, theta), dim=-2)
        raise ValueError("goal_fusion must be dual or mixed")

    def forward(self, x: torch.Tensor, hx=None, return_state: bool = False,
                return_sequence: bool = False):
        if x.ndim != 4 or tuple(x.shape[2:]) != (self.img_h + 2, self.img_w):
            raise ValueError(
                f"Expected [batch,steps,{self.img_h + 2},{self.img_w}], got {tuple(x.shape)}"
            )
        batch, steps = x.shape[:2]
        if steps < 1:
            raise ValueError("At least one observation is required")
        if hx is not None and len(hx) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} spatial states, got {len(hx)}")
        # Apply the encoder independently per observation while keeping (h, w).
        occupancy = x[:, :, :self.img_h, :].reshape(batch * steps, 1, self.img_h, self.img_w)
        encoded = self.encoder(occupancy)
        _, channels, height, width = encoded.shape
        encoded = encoded.reshape(batch, steps, channels, height, width)
        target = x[:, :, -2:, 0]  # [B, T, 2], one (d, theta) per observation
        target_maps = self.make_target_maps(target, height, width, self.goal_fusion)
        fused = torch.cat((encoded, target_maps), dim=2)

        state = list(hx) if hx is not None else [None] * self.num_layers
        outputs = []
        for step in range(steps):
            z = fused[:, step]
            next_state = []
            for layer, cell in enumerate(self.spatial_cells):
                h, c = cell(z, state[layer])
                z = h
                next_state.append((h, c))
            state = next_state
            if return_sequence:
                outputs.append(self.head(z))
        prediction = torch.stack(outputs, dim=1) if return_sequence else self.head(z)
        if return_state:
            return prediction, state
        return prediction
