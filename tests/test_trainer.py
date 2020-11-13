import pytest
import os
import shutil
import pkg_resources
import gc
from transformers import (
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForQuestionAnswering,
    DistilBertTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from kitanaqa.trainer.arguments import ModelArguments
from kitanaqa.trainer.train import Trainer
from kitanaqa.trainer.utils import load_and_cache_examples

MODEL_CLASSES = {
    "albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
    "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
}

TRAIN_PATH = pkg_resources.resource_filename(
            'kitanaqa', 'support/unittest-squad.json')
EVAL_PATH = pkg_resources.resource_filename(
            'kitanaqa', 'support/unittest-squad.json')

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class TrainerTester:
    def __init__(
            self,
            model_type,
            model_name_or_path,
            output_dir,
            cache_dir,
            data_dir,
            train_file_path,
            predict_file_path,
            aug_file_path,
            do_aug,
            do_alum,
            alpha,
            eps,
            eta,
            sigma,
            do_train,
            do_adv_eval,
            do_eval,
            per_device_train_batch_size,
            per_device_eval_batch_size,
            gradient_accumulation_steps,
            eval_all_checkpoints,
            num_train_epochs,
            max_steps,
            save_steps,
            seed,
            fp16):

        args = {
            "model_type": model_type,
            "model_name_or_path": model_name_or_path,
            "output_dir": output_dir,
            "cache_dir": cache_dir,
            "data_dir": data_dir,
            "train_file_path": train_file_path,
            "predict_file_path": predict_file_path,
            "aug_file_path": aug_file_path,
            "do_aug": do_aug,
            "do_alum": do_alum,
            "alpha": alpha,
            "eps": eps,
            "eta": eta,
            "sigma": sigma,
            "do_train": do_train,
            "do_adv_eval": do_adv_eval,
            "do_eval": do_eval,
            "per_device_train_batch_size": per_device_train_batch_size,
            "per_device_eval_batch_size": per_device_eval_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "eval_all_checkpoints": eval_all_checkpoints,
            "num_train_epochs" : num_train_epochs,
            "max_steps": max_steps,
            "save_steps": save_steps,
            "seed": seed,
            "fp16": fp16,
        }
        parser = HfArgumentParser(dataclass_types=[ModelArguments, TrainingArguments])
        self.model_args, self.training_args = parser.parse_dict(args)

        # Load model and tokenizer
        config, self.model_cls, tokenizer_cls = MODEL_CLASSES[self.model_args.model_type]
        self.tokenizer = tokenizer_cls.from_pretrained(
            self.model_args.tokenizer_name_or_path if self.model_args.tokenizer_name_or_path else self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
        )
        model = self.model_cls.from_pretrained(
            self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
        )

        # Load training dataset
        if self.training_args.do_train:
            train_dataset = load_and_cache_examples(self.model_args, self.tokenizer)
        else:
            train_dataset = None

        # Initialize the Trainer
        self.trainer = Trainer(
            model_args=self.model_args,
            data_collator=None,
            model=model,
            tokenizer=self.tokenizer,
            args=self.training_args,
            train_dataset=train_dataset,
            prediction_loss_only=True,
        )

    def _setup_env(self):
        if not os.path.exists(self.training_args.output_dir):
            os.mkdir(self.training_args.output_dir)
        elif os.path.exists(self.training_args.output_dir):
            shutil.rmtree(self.training_args.output_dir)
            os.mkdir(self.training_args.output_dir)

    def _reset_env(self):
        if os.path.exists(self.training_args.output_dir):
            shutil.rmtree(self.training_args.output_dir)
        if os.path.exists("./runs"):
            shutil.rmtree("./runs")

    def _train_and_check_results(self):
        self._setup_env()

        # Training
        self.trainer.train(
            model_path=self.model_args.model_name_or_path if os.path.isdir(self.model_args.model_name_or_path) else None
        )

        model_path = ('/'.join([self.training_args.output_dir, 'checkpoint-1']))
        assert os.path.exists(model_path)

    def _eval_and_check_results(self):
        self._setup_env()

        # Load SQuAD-specific dataset and examples for metric calculation
        dataset, examples, features = load_and_cache_examples(
            self.model_args,
            self.tokenizer,
            evaluate=True,
            output_examples=True)

        for predict_set in examples:
            results = self.trainer.evaluate(
                '/'.join([self.training_args.output_dir, 'checkpoint-1']),
                self.model_args,
                self.tokenizer,
                dataset[predict_set],
                examples[predict_set],
                features[predict_set],
            )

            results_keys = set([
                'exact',
                'f1',
                'total',
            ])
            assert results_keys.issubset(set(results.keys()))
            assert isinstance(results['f1'], float)
            assert isinstance(results['exact'], float)

    def _eval_adv_and_check_results(self):
        self._setup_env()

        # Load SQuAD-specific dataset and examples for metric calculation
        dataset, examples, features = load_and_cache_examples(
            self.model_args,
            self.tokenizer,
            evaluate=True,
            output_examples=True)

        for predict_set in examples:
            results = self.trainer.adv_evaluate(
                '/'.join([self.training_args.output_dir, 'checkpoint-1']),
                self.model_args,
                self.tokenizer,
                dataset[predict_set],
                examples[predict_set],
                features[predict_set],
            )

            results_keys = set([
                'exact',
                'f1',
                'total',
            ])
            assert results_keys.issubset(set(results.keys()))
            assert isinstance(results['f1'], float)
            assert isinstance(results['exact'], float)


def test_alum_train():
        hparams = {
            "model_type" : "distilbert",
            "model_name_or_path" : "distilbert-base-uncased",
            "output_dir" : "./unittest_outputs",
            "cache_dir" : "./unittest_outputs",
            "data_dir" : "./unittest_outputs",
            "train_file_path" : TRAIN_PATH,
            "predict_file_path" : {"dev-v1.1": EVAL_PATH},
            "aug_file_path" : None,
            "do_aug" : False,
            "do_alum" : True,
            "alpha" : 5,
            "eps" : 1e-4,
            "eta" : 1e-5,
            "sigma" : 1e-3,
            "do_train" : True,
            "do_adv_eval" : False,
            "do_eval" : False,
            "per_device_train_batch_size" : 2,
            "per_device_eval_batch_size" : 1,
            "gradient_accumulation_steps" : 2,
            "eval_all_checkpoints" : False,
            "num_train_epochs" : 1,
            "max_steps" : -1,
            "save_steps" : 1,
            "seed" : 512,
            "fp16" : False
        }
        trainer = TrainerTester(**hparams)
        trainer._train_and_check_results()
        del trainer
        gc.collect()


def test_adv_eval():
        hparams = {
            "model_type" : "distilbert",
            "model_name_or_path" : "distilbert-base-uncased",
            "output_dir" : "./unittest_outputs",
            "cache_dir" : "./unittest_outputs",
            "data_dir" : "./unittest_outputs",
            "train_file_path" : TRAIN_PATH,
            "predict_file_path" : {"dev-v1.1": EVAL_PATH},
            "aug_file_path" : None,
            "do_aug" : False,
            "do_alum" : False,
            "alpha" : 5,
            "eps" : 1e-4,
            "eta" : 1e-5,
            "sigma" : 1e-3,
            "do_train" : False,
            "do_adv_eval" : True,
            "do_eval" : True,
            "per_device_train_batch_size" : 2,
            "per_device_eval_batch_size" : 1,
            "gradient_accumulation_steps" : 2,
            "eval_all_checkpoints" : False,
            "num_train_epochs" : 1,
            "max_steps" : -1,
            "save_steps" : 1000,
            "seed" : 512,
            "fp16" : False
        }
        trainer = TrainerTester(**hparams)
        trainer._eval_adv_and_check_results()
        del trainer
        gc.collect()


def test_regular_train():
        hparams = {
            "model_type" : "distilbert",
            "model_name_or_path" : "distilbert-base-uncased",
            "output_dir" : "./unittest_outputs",
            "cache_dir" : "./unittest_outputs",
            "data_dir" : "./unittest_outputs",
            "train_file_path" : TRAIN_PATH,
            "predict_file_path" : {"dev-v1.1": EVAL_PATH},
            "aug_file_path" : None,
            "do_aug" : False,
            "do_alum" : False,
            "alpha" : 5,
            "eps" : 1e-4,
            "eta" : 1e-5,
            "sigma" : 1e-3,
            "do_train" : True,
            "do_adv_eval" : False,
            "do_eval" : False,
            "per_device_train_batch_size" : 2,
            "per_device_eval_batch_size" : 1,
            "gradient_accumulation_steps" : 2,
            "eval_all_checkpoints" : False,
            "num_train_epochs" : 1,
            "max_steps" : -1,
            "save_steps" : 1,
            "seed" : 512,
            "fp16" : False
        }
        trainer = TrainerTester(**hparams)
        trainer._train_and_check_results()
        del trainer
        gc.collect()


def test_regular_eval():
        hparams = {
            "model_type" : "distilbert",
            "model_name_or_path" : "distilbert-base-uncased",
            "output_dir" : "./unittest_outputs",
            "cache_dir" : "./unittest_outputs",
            "data_dir" : "./unittest_outputs",
            "train_file_path" : TRAIN_PATH,
            "predict_file_path" : {"dev-v1.1": EVAL_PATH},
            "aug_file_path" : None,
            "do_aug" : False,
            "do_alum" : False,
            "alpha" : 5,
            "eps" : 1e-4,
            "eta" : 1e-5,
            "sigma" : 1e-3,
            "do_train" : False,
            "do_adv_eval" : False,
            "do_eval" : True,
            "per_device_train_batch_size" : 2,
            "per_device_eval_batch_size" : 1,
            "gradient_accumulation_steps" : 2,
            "eval_all_checkpoints" : False,
            "num_train_epochs" : 1,
            "max_steps" : -1,
            "save_steps" : 1000,
            "seed" : 512,
            "fp16" : False
        }
        trainer = TrainerTester(**hparams)
        trainer._eval_and_check_results()

        trainer._reset_env()
