import dataclasses
from typing import Any, Callable, Mapping, Optional

import numpy as np
import torch as t
import torch.nn as nn
from tqdm import tqdm

import wandb
from experiments import curriculum as curriculum_lib
from experiments import range_evaluation, utils
from tasks import task as task_lib

_LossMetrics = Optional[Mapping[str, t.Tensor]]


@dataclasses.dataclass
class ClassicTrainingParams:
    """Parameters needed to train classical architectures."""

    seed: int  # Used to sample during forward pass (e.g. from final logits).
    model_init_seed: int  # Used to initialize model parameters.
    training_steps: int
    log_frequency: int

    task: task_lib.GeneralizationTask
    length_curriculum: curriculum_lib.Curriculum
    batch_size: int

    model: nn.Module
    loss_fn: Callable[[t.Tensor, t.Tensor], tuple[float, _LossMetrics]]
    learning_rate: float
    test_model: Optional[nn.Module] = None
    max_grad_norm: float = 1.0
    is_autoregressive: bool = False

    compute_full_range_test: bool = False
    range_test_total_batch_size: int = 512
    range_test_sub_batch_size: int = 64
    max_range_test_length: int = 100

    validate_steps: int = 1000
    use_wandb: bool = True

    accuracy_fn: Optional[Callable[[t.Tensor, t.Tensor], t.Tensor]] = None

    # optimizer
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.0


class TrainingWorker:
    """Training worker."""

    def __init__(self, training_params: ClassicTrainingParams, use_tqdm: bool = False):
        self._training_params = training_params
        self._use_tqdm = use_tqdm
        self._is_autoregressive = training_params.is_autoregressive

    def run(
        self,
    ) -> tuple[list[Mapping[str, Any]], Optional[list[Mapping[str, Any]]], t.Tensor]:
        training_params = self._training_params
        results = []
        model = training_params.model
        # device = model.device
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
        model.to(device)

        task = training_params.task
        length_curriculum = training_params.length_curriculum

        loss_fn = training_params.loss_fn
        accuracy_fn = training_params.accuracy_fn

        optimizer = t.optim.AdamW(
            model.parameters(),
            lr=training_params.learning_rate,
            betas=(training_params.adam_beta1, training_params.adam_beta2),
            weight_decay=training_params.weight_decay,
        )

        if self._is_autoregressive:
            raise NotImplementedError
        else:
            pad_sequence = utils.pad_sequence_with_empty_targets(
                generalization_task=task,
                # computation_steps_mult=self._computation_steps_mult,
                # include_eos=training_params.include_eos,
            )

        @t.inference_mode()
        def validate():
            model.eval()
            validation_losses = []
            validation_accuracies = []
            for length, validation_batch in enumerate(training_params.validation_set):
                output_length = validation_batch["output_length"]
                validation_batch_input = validation_batch["input"].to(model.device)
                validation_batch_output = validation_batch["output"].to(model.device)

                output = model(validation_batch_input)

                # reg_loss = None
                if type(output) is dict:
                    # reg_loss = output["reg_loss"]
                    output = output["output"]

                if not training_params.single_output:
                    output = output[:, -output_length:]

                validation_loss, _ = loss_fn(output, validation_batch_output)
                if accuracy_fn is not None:
                    validation_accuracy = accuracy_fn(output, validation_batch_output)
                else:
                    validation_accuracy = None
                    validation_losses.append(float(t.mean(validation_loss)))
                    validation_accuracies.append(float(t.mean(validation_accuracy)))
            return np.mean(validation_losses), np.mean(validation_accuracies)

        steps = range(training_params.training_steps)
        if self._use_tqdm:
            steps = tqdm(steps)

        for step in steps:
            model.train()
            length = length_curriculum.sample_sequence_length(step)
            # output_length = task.output_length(length)
            train_batch = task.sample_batch(length=length, batch_size=training_params.batch_size)
            # if self._is_autoregressive:
            #    raise NotImplementedError
            # else:
            #  train_batch["input"] = pad_sequence(train_batch["input"])
            #    pass

            train_batch_input = train_batch["input"].to(device)
            train_batch_output = train_batch["output"].to(device)

            optimizer.zero_grad(set_to_none=True)

            # print(f"{train_batch_input.shape=}") # [batch_size, length]
            # print(f"{train_batch_output.shape=}") # [batch_size]

            output = model(train_batch_input)

            # print(f"Output: {output}")

            # reg_loss = None
            if type(output) is dict:
                # reg_loss = output["reg_loss"]
                output = output["output"]

            # if not training_params.single_output:
            #    output = output[:, -output_length:]
            train_loss, train_metrics = loss_fn(output, train_batch_output)

            if accuracy_fn is not None:
                train_accuracy = accuracy_fn(output, train_batch_output)
                # print(output)
                # print(train_batch_output)
                # if train_accuracy > best_accuracy:
                #   best_model = model
            else:
                train_accuracy = None

            train_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), training_params.max_grad_norm)
            optimizer.step()

            # if step % training_params.validate_steps == 0 and training_params.validation_set is not None:
            #    validation_loss, validation_accuracy = validate()
            # else:
            #    validation_accuracy = validation_loss = None

            log_freq = training_params.log_frequency
            if (log_freq > 0) and (step % log_freq == 0):
                log_data = {
                    "step": step,
                    "train_loss": float(train_loss),
                }
                if training_params.accuracy_fn is not None:
                    log_data["train_accuracy"] = float(train_accuracy)
                for key, value in train_metrics.items():
                    log_data[".".join(["train_metrics", key])] = np.array(value)
                results.append(log_data)

                if training_params.use_wandb:
                    wandb.log(
                        {
                            "train": {
                                "loss": float(train_loss),
                                "accuracy": float(train_accuracy),
                            },
                        },
                        step=step,
                    )

        eval_results = None
        if training_params.compute_full_range_test:
            eval_params = range_evaluation.EvaluationParams(
                # model=training_params.test_model or model,
                model=model,
                accuracy_fn=training_params.accuracy_fn,
                sample_batch=task.sample_batch,
                max_test_length=training_params.max_range_test_length,
                total_batch_size=training_params.range_test_total_batch_size,
                sub_batch_size=training_params.range_test_sub_batch_size,
                is_autoregressive=training_params.is_autoregressive,
                task=task,
                # computation_steps_mult=self._computation_steps_mult,
                # single_output=training_params.single_output,
                use_wandb=training_params.use_wandb,
                # include_eos=training_params.include_eos,
            )

        writer = self.writer if hasattr(self, "writer") else None
        eval_results = range_evaluation.range_evaluation(eval_params, use_tqdm=False, tboard_writer=writer)

        return results, eval_results
