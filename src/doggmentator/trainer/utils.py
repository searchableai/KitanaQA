import random
import torch
import os
import logging
import requests
import glob
import numpy as np
from typing import Tuple
from torch.utils.data import Dataset
from transformers.data.processors.squad import SquadV1Processor, SquadV2Processor
from transformers import squad_convert_examples_to_features
from prefect import Flow, task
from prefect.utilities.notifications import slack_notifier
from doggmentator.trainer.train import Trainer
from transformers import (
    WEIGHTS_NAME,
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
)

MODEL_CLASSES = {
    "albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
    "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
}

logger = logging.getLogger(__name__)


def custom_scheduler(
        max_step: int,
        update_fn: Callable[int]) -> float:
    """
    Create a custom generator for an input param
    """
    for step in max_step:
        yield update_fn(step)


def get_custom_exp(
        max_steps: int,
        max_val: float,
        min_val: float) -> Iterable:
    """
    Create a custom exponential scheduler
    """
    N0 = max_val
    N1 = np.log(min_val/max_val)/max_steps
    update_fn = lambda x: N0 * np.log(N1 * x)
    return custom_scheduler(max_steps, update_fn)


def get_custom_linear(
        max_steps: int,
        start_val: float,
        end_val: float) -> Iterable:
    """
    Create a custom linear scheduler
    """
    N0 = min(start_val, end_val)
    N1 = max(start_val, end_val)
    N3 = N1/N0
    if start_val > end_val:
        N3 *= -1
    update_fn = lambda x: N3 * x + start_val
    return custom_scheduler(max_steps, update_fn)


def load_and_cache_examples(
        args,
        tokenizer,
        evaluate=False,
        use_aug_path=False,
        output_examples=False) -> torch.utils.data.TensorDataset:
    """
    Load SQuAD-like data features from cache or dataset file
    """

    if not args.train_file_path and not (args.do_aug and args.aug_file_path):
        logging.error('load_and_cache_examples requires one of either \"train_file_path\", \"aug_file_path\"')

    # Use the augmented data or the original training data
    train_or_aug_path = args.train_file_path if not use_aug_path else args.aug_file_path

    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if not args.data_dir and ((evaluate and not args.predict_file_path) or (not evaluate and not train_or_aug_path)):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if args.version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
        else:
            processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
            if evaluate:
                examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file_path)
            else:
                examples = processor.get_train_examples(args.data_dir, filename=train_or_aug_path)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            #threads=args.threads,
        )

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if output_examples:
        return dataset, examples, features
    return dataset


slack_url = os.environ['SLACK_WEBHOOK_URL'] if 'SLACK_WEBHOOK_URL' in os.environ else None
def post_to_slack(obj, old_state, new_state):
    """
    Post a msg to Slack url if configured, else simply return new_state object
    """
    if slack_url:
        if new_state.is_finished():
            msg = "{0} finished in state {1} --- results {2}".format(obj, new_state, new_state.result)

            # replace URL with your Slack webhook URL
            requests.post(slack_url, json={"text": msg})

    return new_state


@task(name="eval", state_handlers=[post_to_slack])
def eval_task(args):
    model_args, training_args = args
    results = {}
    if model_args.eval_all_checkpoints:
        checkpoints = [training_args.output_dir]
        checkpoints = list(
            os.path.dirname(c)
            for c in sorted(glob.glob(training_args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
        )
        checkpoints = [x for x in checkpoints if 'checkpoint' in x]
    else:
        if os.path.exists(model_args.model_name_or_path):
            raise Exception("Must specify parameter model_name_or_path")
        checkpoints = [model_args.model_name_or_path]
    for checkpoint in checkpoints:
        # Load model and tokenizer
        config, model_cls, tokenizer_cls = MODEL_CLASSES[model_args.model_type]
        tokenizer = tokenizer_cls.from_pretrained(
            model_args.tokenizer_name_or_path if model_args.tokenizer_name_or_path else checkpoint,
            cache_dir=model_args.cache_dir,
        )
        model = model_cls.from_pretrained(
            checkpoint,
            cache_dir=model_args.cache_dir,
        )

        # Initialize the Trainer
        trainer = Trainer(
            data_collator=None,
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            prediction_loss_only=True,
        )

        # Load SQuAD-specific dataset and examples for metric calculation
        dataset, examples, features = load_and_cache_examples(
            model_args,
            tokenizer,
            evaluate=True,
            output_examples=True)

        model_idx = checkpoint.split("-")[-1]
        results[model_idx] = {
                                'model_args': model_args,
                                'training_args':training_args,
                                'eval':trainer.evaluate(
                                        checkpoint,
                                        model_args,
                                        tokenizer,
                                        dataset,
                                        examples,
                                        features)
                            }
    return results


@task(name="train", state_handlers=[post_to_slack])
def train_task(args, model, tokenizer, train_dataset):
    model_args, training_args = args

    # Initialize the Trainer
    trainer = Trainer(
        model_args=model_args,
        data_collator=None,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        prediction_loss_only=True,
    )

    # Training
    trainer.train(
        model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
    )
    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_master():
        tokenizer.save_pretrained(training_args.output_dir)


def build_flow(args, label: str='default-train', model=None, tokenizer=None, train_dataset=None):
    model_args, training_args = args
    with Flow(label) as f:
        if training_args.do_eval and training_args.do_train:
            res = eval_task(
                    (model_args, training_args),
                    upstream_tasks=[
                        train_task(
                            (model_args, training_args),
                            model,
                            tokenizer,
                            train_dataset
                        )
                    ]
                )
        elif training_args.do_eval:
            res = eval_task((model_args, training_args))
        elif training_args.do_train:
            res = train_task(
                        (model_args, training_args),
                        model,
                        tokenizer,
                        train_dataset
                    )
        else:
            f = None
            logging.error('Flow must be instantiated with at least one of \"do_train\", \"do_eval\"')
    return f

def is_apex_available():
    try:
        from apex import amp  # noqa: F401
        _has_apex = True
    except ImportError:
        _has_apex = False
    return _has_apex


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
