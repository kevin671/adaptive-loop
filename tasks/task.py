import abc
from typing import TypedDict

import torch as t
import torch.nn.functional as F

Batch = TypedDict("Batch", {"input": t.Tensor, "output": t.Tensor})


class GeneralizationTask(abc.ABC):

    @abc.abstractmethod
    def sample_batch(self, batch_size: int, length: int) -> Batch:
        """Returns a batch of inputs/outputs."""

    def pointwise_loss_fn(self, output: t.Tensor, target: t.Tensor) -> t.Tensor:
        """Returns the pointwise loss between an output and a target."""
        loss = -target * F.log_softmax(output, dim=-1)
        return loss

    def accuracy_fn(self, output: t.Tensor, target: t.Tensor) -> t.Tensor:
        """Returns the accuracy between an output and a target."""
        return t.argmax(output, axis=-1) == t.argmax(target, axis=-1)

    def accuracy_mask(self, target: t.Tensor) -> t.Tensor:
        """Returns a mask to compute the accuracies, to remove the superfluous ones."""
        return t.ones(target.shape[:-1]).to(target.device)

    @property
    @abc.abstractmethod
    def input_size(self) -> int:
        """Returns the size of the input of the models trained on this task."""

    @property
    @abc.abstractmethod
    def output_size(self) -> int:
        """Returns the size of the output of the models trained on this task."""

    def output_length(self, input_length: int) -> int:
        """Returns the length of the output, given an input length."""
        del input_length
        return 1
