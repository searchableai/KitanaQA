import json
import logging
import os
import sys
import numpy as np
import torch
import glob
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from transformers import (
    WEIGHTS_NAME,
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    DistilBertConfig,
    DistilBertForQuestionAnswering,
    DistilBertTokenizer,
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    HfArgumentParser,
    TrainingArguments,
)

from doggmentator.trainer.arguments import ModelArguments
from doggmentator.trainer.train import Trainer
from doggmentator.trainer.utils import set_seed, load_and_cache_examples, is_apex_available, post_to_slack

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
    "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
}


if __name__ == "__main__":

    # Initialize args
    parser = HfArgumentParser(dataclass_types=[ModelArguments, TrainingArguments])
    args_file = "/home/ubuntu/dev/Doggmentator/src/doggmentator/trainer/args.json"
    model_args, training_args = parser.parse_json_file(args_file)

    if model_args.model_type not in list(MODEL_CLASSES.keys()):
        raise NotImplementedError("Model type should be 'bert', 'albert'")
    if not is_apex_available():
        training_args.fp16 = False

    # Setup the environment
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )
    # TODO: check if tmp dirs exist and mkdirs if necessary

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args)

    # Load model and tokenizer
    config, model_cls, tokenizer_cls = MODEL_CLASSES[model_args.model_type]
    tokenizer = tokenizer_cls.from_pretrained(
        model_args.tokenizer_name_or_path if model_args.tokenizer_name_or_path else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = model_cls.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    # Load training dataset
    train_dataset = load_and_cache_examples(model_args, tokenizer)
    
    from prefect import Flow, task
    from prefect.utilities.notifications import slack_notifier

    #@task(name="train")
    @task(name="train", state_handlers=[post_to_slack])
    def train_task(args):
        model_args, training_args = args

        # Initialize the Trainer
        trainer = Trainer(
            model_args=model_args,
            data_collator=None,
            model=model,
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

    # Evaluation
    # Load SQuAD-specific dataset and examples for metric calculation
    dataset, examples, features = load_and_cache_examples(model_args, tokenizer, evaluate=True, output_examples=True)

    #@task(name="eval")
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
                model_args.tokenizer_name_or_path if model_args.tokenizer_name_or_path else model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
            )
            model = model_cls.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
            )

            # Initialize the Trainer
            trainer = Trainer(
                data_collator=None,
                model=model,
                args=training_args,
                prediction_loss_only=True,
            )
    
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

    with Flow('Train-Eval') as f:
        res = eval_task(
                (model_args, training_args),
                upstream_tasks=[
                        train_task((model_args, training_args))
                ]
            )

    f.run()

    # TODO: Log Results

    # TODO: Deploy Model
