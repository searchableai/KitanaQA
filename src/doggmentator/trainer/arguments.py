from dataclasses import dataclass, field
from typing import Dict, List, Optional

def default_logdir() -> str:
    """
    Same default as PyTorch
    """
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join("runs", current_time + "_" + socket.gethostname())

'''
@dataclass
class TrainingArguments:
    """
    TrainingArguments is the subset of the arguments we use in our example scripts
    **which relate to the training loop itself**.
    Using :class:`~transformers.HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on the command line.
    Parameters:
        output_dir (:obj:`str`):
            The output directory where the model predictions and checkpoints will be written.
        overwrite_output_dir (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If :obj:`True`, overwrite the content of the output directory. Use this to continue training if
            :obj:`output_dir` points to a checkpoint directory.
        do_train (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to run training or not.
        do_eval (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to run evaluation on the dev set or not.
        do_predict (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to run predictions on the test set or not.
        evaluate_during_training (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to run evaluation during training at each logging step or not.
        prediction_loss_only (:obj:`bool`, `optional`, defaults to `False`):
            When performing evaluation and predictions, only returns the loss.
        per_device_train_batch_size (:obj:`int`, `optional`, defaults to 8):
            The batch size per GPU/TPU core/CPU for training.
        per_device_eval_batch_size (:obj:`int`, `optional`, defaults to 8):
            The batch size per GPU/TPU core/CPU for evaluation.
        gradient_accumulation_steps: (:obj:`int`, `optional`, defaults to 1):
            Number of updates steps to accumulate the gradients for, before performing a backward/update pass.
        learning_rate (:obj:`float`, `optional`, defaults to 5e-5):
            The initial learning rate for Adam.
        weight_decay (:obj:`float`, `optional`, defaults to 0):
            The weight decay to apply (if not zero).
        adam_epsilon (:obj:`float`, `optional`, defaults to 1e-8):
            Epsilon for the Adam optimizer.
        max_grad_norm (:obj:`float`, `optional`, defaults to 1.0):
            Maximum gradient norm (for gradient clipping).
        num_train_epochs(:obj:`float`, `optional`, defaults to 3.0):
            Total number of training epochs to perform.
        max_steps (:obj:`int`, `optional`, defaults to -1):
            If set to a positive number, the total number of training steps to perform. Overrides
            :obj:`num_train_epochs`.
        warmup_steps (:obj:`int`, `optional`, defaults to 0):
            Number of steps used for a linear warmup from 0 to :obj:`learning_rate`.
        logging_dir (:obj:`str`, `optional`):
            Tensorboard log directory. Will default to `runs/**CURRENT_DATETIME_HOSTNAME**`.
        logging_first_step (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Wheter to log and evalulate the first :obj:`global_step` or not.
        logging_steps (:obj:`int`, `optional`, defaults to 500):
            Number of update steps between two logs.
        save_steps (:obj:`int`, `optional`, defaults to 500):
            Number of updates steps before two checkpoint saves.
        save_total_limit (:obj:`int`, `optional`):
            If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in
            :obj:`output_dir`.
        no_cuda (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to not use CUDA even when it is available or not.
        seed (:obj:`int`, `optional`, defaults to 42):
            Random seed for initialization.
        fp16 (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use 16-bit (mixed) precision training (through NVIDIA apex) instead of 32-bit training.
        fp16_opt_level (:obj:`str`, `optional`, defaults to 'O1'):
            For :obj:`fp16` training, apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details
            on the `apex documentation <https://nvidia.github.io/apex/amp.html>`__.
        local_rank (:obj:`int`, `optional`, defaults to -1):
            During distributed training, the rank of the process.
        tpu_num_cores (:obj:`int`, `optional`):
            When training on TPU, the mumber of TPU cores (automatically passed by launcher script).
        debug (:obj:`bool`, `optional`, defaults to :obj:`False`):
            When training on TPU, whether to print debug metrics or not.
        dataloader_drop_last (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to drop the last incomplete batch (if the length of the dataset is not divisible by the batch size)
            or not.
        eval_steps (:obj:`int`, `optional`, defaults to 1000):
            Number of update steps between two evaluations.
        past_index (:obj:`int`, `optional`, defaults to -1):
            Some models like :doc:`TransformerXL <../model_doc/transformerxl>` or :doc`XLNet <../model_doc/xlnet>` can
            make use of the past hidden states for their predictions. If this argument is set to a positive int, the
            ``Trainer`` will use the corresponding output (usually index 2) as the past state and feed it to the model
            at the next training step under the keyword argument ``mems``.
        run_name (:obj:`str`, `optional`):
            A descriptor for the run. Notably used for wandb logging.
    """

    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory."
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )

    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    evaluate_during_training: bool = field(
        default=False, metadata={"help": "Run evaluation during training at each logging step."},
    )
    prediction_loss_only: bool = field(
        default=False, metadata={"help": "When performing evaluation and predictions, only returns the loss."},
    )

    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )

    per_gpu_train_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Deprecated, the use of `--per_device_train_batch_size` is preferred. "
            "Batch size per GPU/TPU core/CPU for training."
        },
    )
    per_gpu_eval_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Deprecated, the use of `--per_device_eval_batch_size` is preferred."
            "Batch size per GPU/TPU core/CPU for evaluation."
        },
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )

    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for Adam."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for Adam optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for Adam optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for Adam optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})

    logging_dir: Optional[str] = field(default_factory=default_logdir, metadata={"help": "Tensorboard log dir."})
    logging_first_step: bool = field(default=False, metadata={"help": "Log and eval the first global_step"})
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Limit the total amount of checkpoints."
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            )
        },
    )
    no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available"})
    seed: int = field(default=42, metadata={"help": "random seed for initialization"})

    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"},
    )
    fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help": (
                "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                "See details at https://nvidia.github.io/apex/amp.html"
            )
        },
    )
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})

    tpu_num_cores: Optional[int] = field(
        default=None, metadata={"help": "TPU: Number of TPU cores (automatically passed by launcher script)"}
    )
    tpu_metrics_debug: bool = field(
        default=False,
        metadata={"help": "Deprecated, the use of `--debug` is preferred. TPU: Whether to print debug metrics"},
    )
    debug: bool = field(default=False, metadata={"help": "Whether to print debug metrics on TPU"})

    dataloader_drop_last: bool = field(
        default=False, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )
    eval_steps: int = field(default=1000, metadata={"help": "Run an evaluation every X steps."})

    past_index: int = field(
        default=-1,
        metadata={"help": "If >=0, uses the corresponding part of the output as the past state for next step."},
    )

    run_name: Optional[str] = field(
        default=None, metadata={"help": "An optional descriptor for the run. Notably used for wandb logging."}
    )
'''


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    train_file_path: str = field(
        metadata={"help": "Path for train dataset"},
    )
    predict_file_path: Dict[str, str] = field(
        metadata={"help": "Path for cached eval dataset"},
    )
    model_type: str = field(metadata={"help": "One of 'bert', 'albert'"})
    aug_file_path: str = field(
        default=None, metadata={"help": "Path for augmented train dataset"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    do_lower_case: Optional[str] = field(
        default=True, metadata={"help": "lowercase input text"}
    )
    version_2_with_negative: bool = field(
        default=False,
        metadata={"help": "SQuAD v2.0 training"}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={"help": "SQuAD v2.0 null threshold"}
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "Number of answers to evaluate"}
    )
    verbose_logging: bool = field(
        default=False,
        metadata={"help": "Logging for eval"}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite cached data on load"}
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={"help": "Max length for the input tokens"},
    )
    max_query_length: Optional[int] = field(
        default=64,
        metadata={"help": "Max length for the query segment"}
    )
    max_answer_length: Optional[int] = field(
        default=30,
        metadata={"help": "Max length of answer span tokens"}
    )
    doc_stride: Optional[int] = field(
        default=128
    )
    freeze_embeds: bool = field(
        default=False,
        metadata={"help": "Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."}
    )
    do_aug: bool = field(
        default=True,
        metadata={"help": "Use Augmented training."}
    )
    do_adv_eval: bool = field(
        default=False,
        metadata={"help": "Perform adversarial attack evaluation."}
    )
    do_alum: bool = field(
        default=True,
        metadata={"help": "Use ALUM loss."}
    )
    eta: Optional[float] = field(
        default=1e-3,
        metadata={"help": "Perturbation step size in ALUM training."}
    )
    alpha: Optional[float] = field(
        default=1,
        metadata={"help": "ALUM loss weight."}
    )
    alpha_final: Optional[float] = field(
        default=None,
        metadata={"help": "ALUM final value of loss weight."}
    )
    alpha_schedule: Optional[str] = field(
        default=None,
        metadata={"help": "ALUM alpha param schedule type."}
    )
    eps: Optional[float] = field(
        default=1e-5,
        metadata={"help": "Perturbation radius in ALUM training."}
    )
    sigma: Optional[float] = field(
        default=1e-5,
        metadata={"help": "Variance for normal dist perturbation sampling in ALUM training."}
    )
    K: Optional[float] = field(
        default=1,
        metadata={"help": "Attack approx iteration in ALUM training."}
    )
    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path for data files"},
    )
    eval_all_checkpoints: Optional[str] = field(
        default=False,
        metadata={"help" : "Evaluate all checkpoints in output_dir"}
    )
