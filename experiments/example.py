import numpy as np
import torch
from absl import app, flags

from experiments import constants
from experiments import curriculum as curriculum_lib
from experiments import training

_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size",
    default=128,
    help="Training batch size.",
    lower_bound=1,
)

_SEQUENCE_LENGTH = flags.DEFINE_integer(
    "sequence_length",
    default=40,
    help="Maximum training sequence length.",
    lower_bound=1,
)

_TASK = flags.DEFINE_string(
    "task",
    default="parity_check",
    help="Length generalization task (see `constants.py` for other tasks).",
)

_ARCHITECTURE = flags.DEFINE_string(
    "architecture",
    default="rnn",
    help="Model architecture (see `constants.py` for other architectures).",
)

_IS_AUTOREGRESSIVE = flags.DEFINE_boolean(
    "is_autoregressive",
    default=False,
    help="Whether to use autoregressive sampling or not.",
)

_COMPUTATION_STEPS_MULT = flags.DEFINE_integer(
    "computation_steps_mult",
    default=0,
    help=(
        "The amount of computation tokens to append to the input tape (defined"
        " as a multiple of the input length)"
    ),
    lower_bound=0,
)

_SEED = flags.DEFINE_integer("seed", default=None, help="random seed.")

_LEARNING_RATE = flags.DEFINE_float("lr", default=1e-5, help="learning rate")

_WEIGHT_DECAY = flags.DEFINE_float("wd", default=0.0, help="weight decay.")

_ADAM_BETA1_INV = flags.DEFINE_float(
    "adam_beta1_inv", default=0.1, help="adam_beta1=1-adam_beta1_inv"
)

_TRAIN_STEPS = flags.DEFINE_integer(
    "train_steps", default=0, help="number of training steps."
)

_TEST_LENGTH = flags.DEFINE_integer(
    "test_length", default=500, help="maximum test sequence length."
)

_NUM_LAYERS = flags.DEFINE_integer("num_layers", default=None, help="number of layers")

_EMBED_SIZE = flags.DEFINE_integer("embed_size", default=32, help="embedding size")


def main(unused_argv) -> None:
    # Create the task.
    curriculum = curriculum_lib.UniformCurriculum(
        values=list(range(1, _SEQUENCE_LENGTH.value + 1))
    )
    task = constants.TASK_BUILDERS[_TASK.value]()

    _ARCHITECTURE_PARAMS = {}
    for arg in flags.FLAGS:
        _ARCHITECTURE_PARAMS[str(arg)] = flags.FLAGS[str(arg)]._value
    # Create the model.
    model = constants.MODEL_BUILDERS[_ARCHITECTURE.value](
        output_size=task.output_size,
        return_all_outputs=True,
        **_ARCHITECTURE_PARAMS,
    )

    # Create the loss and accuracy based on the pointwise ones.
    def loss_fn(output, target):
        loss = torch.mean(torch.sum(task.pointwise_loss_fn(output, target), axis=-1))
        # print(f"Loss: {loss}")
        return loss, {}

    def accuracy_fn(output, target):
        mask = task.accuracy_mask(target)
        return torch.sum(mask * task.accuracy_fn(output, target)) / torch.sum(mask)

    # Create the final training parameters.
    training_params = training.ClassicTrainingParams(
        seed=0,
        model_init_seed=0,
        training_steps=10_000,
        log_frequency=100,
        length_curriculum=curriculum,
        batch_size=_BATCH_SIZE.value,
        task=task,
        model=model,
        loss_fn=loss_fn,
        learning_rate=1e-3,
        accuracy_fn=accuracy_fn,
        compute_full_range_test=True,
        max_range_test_length=100,
        range_test_total_batch_size=512,
        range_test_sub_batch_size=64,
        is_autoregressive=_IS_AUTOREGRESSIVE.value,
    )

    training_worker = training.TrainingWorker(
        training_params,
        use_tqdm=True,
        is_autoregressive=_IS_AUTOREGRESSIVE.value,
        computation_steps_mult=_COMPUTATION_STEPS_MULT.value,
    )
    train_results, eval_results = training_worker.run()
    print(train_results)

    # Gather results and print final score.
    accuracies = [r["accuracy"] for r in eval_results]
    score = np.mean(accuracies[_SEQUENCE_LENGTH.value + 1 :])
    print(f"Network score: {score}")


if __name__ == "__main__":
    app.run(main)
