# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, InitVar
from pathlib import Path
from typing import Any, ClassVar

import h5py
import numpy as np
import torch
from torch import nn

from emg2qwerty.charset import CharacterSet, charset
from emg2qwerty.transforms import ToTensor, Transform

@dataclass
class WindowedEMGDataset_v2(torch.utils.data.Dataset):
    """A `torch.utils.data.Dataset` corresponding to an instance of `EMGSessionData`
    that iterates over EMG windows of configurable length and stride.

    Args:
        hdf5_path (str): Path to the session file in hdf5 format.
        window_length (int): Size of each window. Specify None for no windowing
            in which case this will be a dataset of length 1 containing the
            entire session. (default: ``None``)
        stride (int): Stride between consecutive windows. Specify None to set
            this to window_length, in which case there will be no overlap
            between consecutive windows. (default: ``window_length``)
        padding (tuple[int, int]): Left and right contextual padding for
            windows in terms of number of raw EMG samples.
        jitter (bool): If True, randomly jitter the offset of each window.
            Use this for training time variability. (default: ``False``)
        transform (Callable): A composed sequence of transforms that takes
            a window/slice of `EMGSessionData` in the form of a numpy
            structured array and returns a `torch.Tensor` instance.
            (default: ``emg2qwerty.transforms.ToTensor()``)
    """

    hdf5_path: Path
    window_length: InitVar[int | None] = None
    stride: InitVar[int | None] = None
    padding: InitVar[tuple[int, int]] = (0, 0)
    jitter: bool = False
    transform: Transform[np.ndarray, torch.Tensor] = field(default_factory=ToTensor)

    def __post_init__(
        self,
        window_length: int | None,
        stride: int | None,
        padding: tuple[int, int],
    ) -> None:
        with EMGSessionData(self.hdf5_path) as session:
            assert (
                session.condition == "on_keyboard"
            ), f"Unsupported condition {self.session.condition}"
            self.session_length = len(session)

        self.window_length = (
            window_length if window_length is not None else self.session_length
        )
        self.stride = stride if stride is not None else self.window_length
        assert self.window_length > 0 and self.stride > 0

        (self.left_padding, self.right_padding) = padding
        assert self.left_padding >= 0 and self.right_padding >= 0

    def __len__(self) -> int:
        return int(max(self.session_length - self.window_length, 0) // self.stride + 1)

    def __getitem__(self, idx: int) -> tuple[torch.nested.Tensor, torch.Tensor]:
        # Lazy init `EMGSessionData` per dataloading worker
        # since `h5py.File` objects can't be picked.
        if not hasattr(self, "session"):
            self.session = EMGSessionData(self.hdf5_path)

        offset = idx * self.stride

        # Randomly jitter the window offset.
        leftover = len(self.session) - (offset + self.window_length)
        if leftover < 0:
            raise IndexError(f"Index {idx} out of bounds")
        if leftover > 0 and self.jitter:
            offset += np.random.randint(0, min(self.stride, leftover))

        # Expand window to include contextual padding and fetch.
        window_start = max(offset - self.left_padding, 0)
        window_end = offset + self.window_length + self.right_padding
        window = self.session[window_start:window_end]

        # Extract EMG tensor corresponding to the window.
        emg = self.transform(window)
        to_tensor = ToTensor()
        raw = to_tensor(window).unsqueeze(4) # create extra dim to match emg dims
        assert torch.is_tensor(emg)
        assert torch.is_tensor(raw)

        emg = torch.nested.nested_tensor([emg, raw])

        # Extract labels corresponding to the original (un-padded) window.
        timestamps = window[EMGSessionData.TIMESTAMPS]
        start_t = timestamps[offset - window_start]
        end_t = timestamps[(offset + self.window_length - 1) - window_start]
        label_data = self.session.ground_truth(start_t, end_t)
        labels = torch.as_tensor(label_data.labels)

        return emg, labels

    @staticmethod
    def collate(
        samples: Sequence[tuple[torch.nested.Tensor, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Collates a list of samples into a padded batch of inputs and targets.
        Each input sample in the list should be a tuple of (input, target) tensors.
        Also returns the lengths of unpadded inputs and targets for use in loss
        functions such as CTC or RNN-T.

        Follows time-first format. That is, the retured batch is of shape (T, N, ...).
        """
        inputs_spec = [sample[0][0] for sample in samples]  # [(T, ...)]
        inputs_wave = [sample[0][1] for sample in samples]  # [(T, ...)]
        targets = [sample[1] for sample in samples]  # [(T,)]

        # Batch of inputs and targets padded along time
        input_spec_batch = nn.utils.rnn.pad_sequence(inputs_spec)  # (T, N, ...)
        input_wave_batch = nn.utils.rnn.pad_sequence(inputs_wave).squeeze(4)  # (T, N, ...) / remove extra dimension
        target_batch = nn.utils.rnn.pad_sequence(targets)  # (T, N)

        # Lengths of unpadded input and target sequences for each batch entry
        input_spec_lengths = torch.as_tensor(
            [len(_input) for _input in inputs_spec], dtype=torch.int32
        )

        input_wave_lengths = torch.as_tensor(
            [len(_input) for _input in inputs_wave], dtype=torch.int32
        )

        target_lengths = torch.as_tensor(
            [len(target) for target in targets], dtype=torch.int32
        )

        return {
            "inputs_spec": input_spec_batch,
            "inputs_wave": input_wave_batch,
            "targets": target_batch,
            "input_spec_lengths": input_spec_lengths,
            "input_wave_lengths": input_wave_lengths,
            "target_lengths": target_lengths,
        }
