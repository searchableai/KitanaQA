import json
import logging
import os
import sys
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from transformers.file_utils import is_apex_available

from transformers import (
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
from doggmentator.trainer.utils import set_seed, load_and_cache_examples, setup_devices

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
    "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
}

if __name__ == "__main__":

    # Initialize args
    parser = HfArgumentParser(dataclass_types=[ModelArguments, TrainingArguments])
    args_file = "/home/ubuntu/dev/bootcamp/finetune/SQuAD/train/refactored/run/args.json"
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
    
    # Initialize the Trainer
    trainer = Trainer(
        do_alum=model_args.do_alum,
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

    # TODO: Model Selection && Re-init Trainer
    '''
    trainer = Trainer(
        data_collator=None,
        model=model,
        args=training_args,
        prediction_loss_only=True,
    )
    '''

    # Evaluation
    # Load SQuAD-specific dataset and examples for metric calculation
    dataset, examples, features = load_and_cache_examples(model_args, tokenizer, evaluate=True, output_examples=True)
    eval_results = trainer.evaluate(model_args.model_name_or_path, dataset, examples, features)

    # TODO: Log Results

    # TODO: Deploy Model
