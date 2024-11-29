from typing import Callable

import torch as t
import torch.nn.functional as F

from tasks import task

COMPUTATION_EMPTY_TOKEN = 0
OUTPUT_EMPTY_TOKEN = 1


def pad_sequence_with_empty_targets(
    generalization_task: task.GeneralizationTask,
    computation_steps_mult: int = 0,
    include_eos: bool = False,  # True,
) -> Callable[[t.Tensor], t.Tensor]:
    """Pads the inputs to match the output length.

    For a given input tape `input_tape` of vocabulary size `vocab_size`, the
    tape will be transformed to the format
    [`input_tape`, `empty_tape`], where the empty tape token is `vocab_size + 1`.
    The `empty_tape` has the same length as the task output.

    Args:
      generalization_task: The task that we train on.
      computation_steps_mult: The amount of empty cells to append to the input
        tape. This variable is a multiplier and the actual number of cells is
        `computation_steps_mult * input_length`.
      single_output: Whether to return the squeezed tensor of values.
    """

    def padded_sequence(x: t.Tensor) -> t.Tensor:
        batch_size, input_length, input_size = x.shape
        output_length = generalization_task.output_length(input_length)
        extra_dims_onehot = 1 + int(computation_steps_mult > 0)
        final_input_size = input_size + extra_dims_onehot

        # Add trailing zeros to account for new final_input_size.
        extra_zeros_x = t.zeros((batch_size, input_length, final_input_size - input_size))
        x = t.cat([x, extra_zeros_x], dim=-1)

        if not include_eos:
            return x

        computation_tape = t.full(
            (batch_size, computation_steps_mult * input_length),
            fill_value=input_size + COMPUTATION_EMPTY_TOKEN,
        )
        computation_tape = F.one_hot(computation_tape, num_classes=final_input_size)

        output_tokens = t.full(
            (batch_size, output_length),
            fill_value=input_size + OUTPUT_EMPTY_TOKEN - int(computation_steps_mult == 0),
        )
        output_tokens = F.one_hot(output_tokens, num_classes=final_input_size)
        final_input = t.cat([x, computation_tape, output_tokens], dim=1)

        return final_input

    return padded_sequence
