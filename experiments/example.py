import numpy as np
import torch
import torch.nn.functional as F
from absl import app, flags

import wandb
from experiments import constants
from experiments import curriculum as curriculum_lib
from experiments import training

_TASK = flags.DEFINE_string("task", default="parity_check", help="task")
_SEQUENCE_LENGTH = flags.DEFINE_integer("sequence_length", default=40, help="Maximum training sequence length.")

_ARCHITECTURE = flags.DEFINE_string("architecture", default="rnn", help="architecture")
_IS_AUTOREGRESSIVE = flags.DEFINE_boolean("is_autoregressive", default=False, help="")
_IS_CAUSAL = flags.DEFINE_integer("is_causal", default=0, help="Whether to use causal attention or not.")
_NUM_LAYERS = flags.DEFINE_integer("num_layers", default=None, help="Number of layers.")
_EMBED_SIZE = flags.DEFINE_integer("embed_size", default=32, help="Embedding size.")
_NUM_HEADS = flags.DEFINE_integer("num_heads", default=None, help="Number of heads.")
_NUM_LOOPS = flags.DEFINE_integer("num_loops", default=None, help="Number of loops.")

_SEED = flags.DEFINE_integer("seed", default=None, help="Random seed.")
_LEARNING_RATE = flags.DEFINE_float("lr", default=1e-3, help="Learning rate.")
_ADAM_BETA1 = flags.DEFINE_float("adam_beta1", default=0.9, help="Adam beta1.")
_ADAM_BETA2 = flags.DEFINE_float("adam_beta2", default=0.999, help="Adam beta2.")
_WEIGHT_DECAY = flags.DEFINE_float("weight_decay", default=0.0, help="Weight decay.")
_DROP_OUT = flags.DEFINE_float("dropout", default=0.1, help="Dropout rate.")
_BATCH_SIZE = flags.DEFINE_integer("batch_size", default=32, help="Batch size.")
_TRAIN_STEPS = flags.DEFINE_integer("train_steps", default=10_000, help="number of training steps.")
_TEST_LENGTH = flags.DEFINE_integer("test_length", default=100, help="maximum test sequence length.")
_WANDB_PROJECT = flags.DEFINE_string("wandb_project", default="chomsky", help="The wandb project")


def main(unused_argv) -> None:
    # Create the task.
    curriculum = curriculum_lib.UniformCurriculum(values=list(range(1, _SEQUENCE_LENGTH.value + 1)))
    task = constants.TASK_BUILDERS[_TASK.value]()

    _ARCHITECTURE_PARAMS = {}
    # for arg in flags.FLAGS:
    #    _ARCHITECTURE_PARAMS[str(arg)] = flags.FLAGS[str(arg)]._value

    _ARCHITECTURE_PARAMS["num_layers"] = _NUM_LAYERS.value
    _ARCHITECTURE_PARAMS["embed_size"] = _EMBED_SIZE.value
    _ARCHITECTURE_PARAMS["hidden_size"] = _EMBED_SIZE.value

    # Create the model.
    model = constants.MODEL_BUILDERS[_ARCHITECTURE.value](
        output_size=task.output_size,
        # return_all_outputs=True,
        input_size=task.input_size,
        **_ARCHITECTURE_PARAMS,
    )

    # Create the loss and accuracy based on the pointwise ones.
    def loss_fn(output, target):
        # loss = torch.mean(torch.sum(task.pointwise_loss_fn(output, target), axis=-1))
        # print(f"Loss: {loss}")
        loss = F.cross_entropy(output, target)
        return loss, {}

    def accuracy_fn(output, target):
        # mask = task.accuracy_mask(target)
        # return torch.sum(mask * task.accuracy_fn(output, target)) / torch.sum(mask)
        accuracy = torch.mean((torch.argmax(output, axis=-1) == target).float())
        return accuracy

    wandb.init(project=_WANDB_PROJECT.value, config=_ARCHITECTURE_PARAMS)

    # Create the final training parameters.
    training_params = training.ClassicTrainingParams(
        seed=0,
        model_init_seed=0,
        training_steps=_TRAIN_STEPS.value,  # 10_000,
        log_frequency=100,
        length_curriculum=curriculum,
        batch_size=_BATCH_SIZE.value,
        task=task,
        model=model,
        loss_fn=loss_fn,
        learning_rate=_LEARNING_RATE.value,  # 1e-3,
        adam_beta1=_ADAM_BETA1.value,
        adam_beta2=_ADAM_BETA2.value,
        weight_decay=_WEIGHT_DECAY.value,
        accuracy_fn=accuracy_fn,
        compute_full_range_test=True,
        max_range_test_length=_TEST_LENGTH.value,  # 100,
        range_test_total_batch_size=512,
        range_test_sub_batch_size=64,
        is_autoregressive=_IS_AUTOREGRESSIVE.value,
    )

    training_worker = training.TrainingWorker(
        training_params,
        use_tqdm=True,
        # is_autoregressive=_IS_AUTOREGRESSIVE.value,
        # computation_steps_mult=_COMPUTATION_STEPS_MULT.value,
    )
    train_results, eval_results = training_worker.run()
    print(train_results)

    # Gather results and print final score.
    accuracies = [r["accuracy"] for r in eval_results]
    score = np.mean(accuracies[_SEQUENCE_LENGTH.value + 1 :])
    print(f"Network score: {score}")


if __name__ == "__main__":
    app.run(main)
