import torch
from torch import nn
from torch.utils.data import DistributedSampler, RandomSampler

import logging
import os
import sys
import copy
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    MBartTokenizer,
    TrainingArguments,
    set_seed,
    Trainer,
)

from transformers_.examples.seq2seq.utils import (
    Seq2SeqDataCollator,
    Seq2SeqDataset,
    assert_all_frozen,
    build_compute_metrics_fn,
    check_output_dir,
    freeze_embeds,
    freeze_params,
    lmap,
    save_json,
    use_task_specific_params,
    write_txt_file,
)
# from transformers_.examples.seq2seq import seq2seq_trainer
from seq2seq_trainer import Seq2SeqTrainer, arg_to_scheduler_choices

from transformers.trainer_utils import EvaluationStrategy
from transformers.optimization import (
    Adafactor,
    AdamW,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DistributedSampler, RandomSampler

from transformers import PreTrainedModel, Trainer, logging
from transformers.configuration_fsmt import FSMTConfig
from transformers.file_utils import is_torch_tpu_available

try:
    from .utils import label_smoothed_nll_loss
except ImportError:
    from utils import label_smoothed_nll_loss

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule,
    "constant_w_warmup": get_constant_schedule_with_warmup,
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())

class Seq2SeqTrainingArguments(TrainingArguments):
    """
    Parameters:
        label_smoothing (:obj:`float`, `optional`, defaults to 0):
            The label smoothing epsilon to apply (if not zero).
        sortish_sampler (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to SortishSamler or not. It sorts the inputs according to lenghts in-order to minimizing the padding size.
        predict_with_generate (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use generate to calculate generative metrics (ROUGE, BLEU).
    """

    label_smoothing: Optional[float] = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (if not zero)."}
    )
    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to SortishSamler or not."})
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    adafactor: bool = field(default=False, metadata={"help": "whether to use adafactor"})
    encoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Encoder layer dropout probability. Goes into model.config."}
    )
    decoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Decoder layer dropout probability. Goes into model.config."}
    )
    dropout: Optional[float] = field(default=None, metadata={"help": "Dropout probability. Goes into model.config."})
    attention_dropout: Optional[float] = field(
        default=None, metadata={"help": "Attention dropout probability. Goes into model.config."}
    )
    lr_scheduler: Optional[str] = field(
        default="linear", metadata={"help": f"Which lr scheduler to use. Selected in {arg_to_scheduler_choices}"}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    freeze_encoder: bool = field(default=False, metadata={"help": "Whether tp freeze the encoder."})
    freeze_embeds: bool = field(default=False, metadata={"help": "Whether  to freeze the embeddings."})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=256,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=256,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    test_max_target_length: Optional[int] = field(
        default=256,
        metadata={
            "help": "The maximum total sequence length for test target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    n_train: Optional[int] = field(default=-1, metadata={"help": "# training examples. -1 means use all."})
    n_val: Optional[int] = field(default=200, metadata={"help": "# validation examples. -1 means use all."})
    n_test: Optional[int] = field(default=1000, metadata={"help": "# test examples. -1 means use all."})
    src_lang: Optional[str] = field(default=None, metadata={"help": "Source language id for translation."})
    tgt_lang: Optional[str] = field(default=None, metadata={"help": "Target language id for translation."})
    eval_beams: Optional[int] = field(default=None, metadata={"help": "# num_beams to use for evaluation."})
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined."},
    )

class E2E_Trainer(Seq2SeqTrainer):
    def __init__(self, config=None, data_args=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if config is None:
            assert isinstance(
                self.model, PreTrainedModel
            ), f"If no `config` is passed the model to be trained has to be of type `PreTrainedModel`, but is {self.model.__class__}"
            self.config = self._actual_model(self.model).config
        else:
            self.config = config

        self.data_args = data_args
        self.vocab_size = self.config.tgt_vocab_size if isinstance(self.config,
                                                                   FSMTConfig) else self.config.vocab_size

        if self.args.label_smoothing != 0 or (
                self.data_args is not None and self.data_args.ignore_pad_token_for_loss):
            assert (
                    self.config.pad_token_id is not None
            ), "Make sure that `config.pad_token_id` is correcly defined when ignoring `pad_token` for loss calculation or doing label smoothing."

        if self.config.pad_token_id is None and self.config.eos_token_id is not None:
            print(
                f"The `config.pad_token_id` is `None`. Using `config.eos_token_id` = {self.config.eos_token_id} for padding.."
            )

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            if self.args.adafactor:
                self.optimizer = Adafactor(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    scale_parameter=False,
                    relative_step=False,
                )

            else:
                self.optimizer = AdamW(
                    optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon
                )

        if self.lr_scheduler is None:
            self.lr_scheduler = self._get_lr_scheduler(num_training_steps)
        else:  # ignoring --lr_scheduler
            print("scheduler is passed to `Seq2SeqTrainer`, `--lr_scheduler` arg is ignored.")

    def _get_lr_scheduler(self, num_training_steps):
        schedule_func = arg_to_scheduler[self.args.lr_scheduler.default]
        if self.args.lr_scheduler == "constant":
            scheduler = schedule_func(self.optimizer)
        elif self.args.lr_scheduler == "constant_w_warmup":
            scheduler = schedule_func(self.optimizer, num_warmup_steps=self.args.warmup_steps)
        else:
            scheduler = schedule_func(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )
        return scheduler

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            return None
        elif is_torch_tpu_available():
            return get_tpu_sampler(self.train_dataset)
        else:
            if self.args.sortish_sampler:
                self.train_dataset.make_sortish_sampler(
                    self.args.per_device_train_batch_size,
                    distributed=self.args.n_gpu > 1
                )

            return (
                RandomSampler(self.train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )

    def _compute_loss(self, model, inputs, labels):
        if self.args.label_smoothing == 0:
            if self.data_args is not None and self.data_args.ignore_pad_token_for_loss:
                # force training to ignore pad token
                logits = model(**inputs, use_cache=False)[0]

                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
                loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            else:
                # compute usual loss via models
                loss, logits = model(**inputs, labels=labels, use_cache=False)[:2]
        else:
            # compute label smoothed loss
            logits = model(**inputs, use_cache=False)[0]
            lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            loss, _ = label_smoothed_nll_loss(
                lprobs, labels, self.args.label_smoothing.default,
                ignore_index=self.config.pad_token_id
            )
        return loss, logits

    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        loss, _ = self._compute_loss(model, inputs, labels)
        return loss

    def prediction_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            A tuple with the loss, logits and labels (each being optional).
        """
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = {
            "max_length": self.data_args.val_max_target_length
            if self.data_args is not None
            else self.config.max_length,
            "num_beams": self.data_args.eval_beams if self.data_args is not None else self.config.num_beams,
        }

        if self.args.predict_with_generate and not self.args.prediction_loss_only:
            generated_tokens = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs,
            )
            # in case the batch is shorter than max length, the output should be padded
            if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
                generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        labels = inputs.pop("labels")
        with torch.no_grad():
            # compute loss on predict data
            loss, logits = self._compute_loss(model, inputs, labels)

        loss = loss.mean().detach()
        if self.args.prediction_loss_only:
            return (loss, None, None)

        logits = generated_tokens if self.args.predict_with_generate else logits

        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return (loss, logits, labels)


def main():
    model_args = ModelArguments(
                                model_name_or_path="google/pegasus-xsum",
                                cache_dir='/cache',
                                )
    data_args = DataTrainingArguments(data_dir="/data"
                                      )
    training_args = Seq2SeqTrainingArguments(output_dir="/model",
                                      do_train=True,
                                      do_eval=True,
                                      per_device_train_batch_size=4,
                                      per_device_eval_batch_size=4,
                                      evaluate_during_training=True,
                                      learning_rate=3e-5,
                                      num_train_epochs=1,
                                      )

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_prefix_space=False,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        # from_tf=".ckpt" in model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # set num_beams for evaluation
    if data_args.eval_beams is None:
        data_args.eval_beams = model.config.num_beams

    if model_args.freeze_embeds:
        freeze_embeds(model)
    if model_args.freeze_encoder:
        freeze_params(model.get_encoder())
        assert_all_frozen(model.get_encoder())

    dataset_class = Seq2SeqDataset

    # Get datasets
    train_dataset = (
        dataset_class(
            tokenizer,
            type_path="train",
            data_dir=data_args.data_dir,
            n_obs=data_args.n_train,
            max_target_length=data_args.max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=model.config.prefix or "",
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        dataset_class(
            tokenizer,
            type_path="val",
            data_dir=data_args.data_dir,
            n_obs=data_args.n_val,
            max_target_length=data_args.val_max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=model.config.prefix or "",
        )
        if training_args.do_eval or training_args.evaluation_strategy != trainer_utils.EvaluationStrategy.NO
        else None
    )
    test_dataset = (
        dataset_class(
            tokenizer,
            type_path="test",
            data_dir=data_args.data_dir,
            n_obs=data_args.n_test,
            max_target_length=data_args.test_max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=model.config.prefix or "",
        )
        if training_args.do_predict
        else None
    )

    # Initialize our Trainer
    compute_metrics_fn = (
        # build_compute_metrics_fn(data_args.task, tokenizer) if training_args.predict_with_generate else
         None
    )

    special_tokens_dict = {'sep_token': '<sep>', 'eos_token': '<eos>', 'bos_token': '<sos>'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    trainer = E2E_Trainer(
        model=model,
        config=config,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Seq2SeqDataCollator(tokenizer, data_args, training_args.tpu_num_cores),
        compute_metrics=compute_metrics_fn,
        data_args=data_args
    )

    # Training
    trainer.train(
        model_path = "/content/drive/My Drive/E2E_T5/model/checkpoint-3000"
        # model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
    )
    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_process_zero():
      trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))
      tokenizer.save_pretrained(training_args.output_dir)


