import torch
import os
import logging
import timeit
import itertools
import numpy as np
from tqdm import tqdm
from typing import Optional
from operator import itemgetter

from torch import nn
from torch.utils.data import SequentialSampler, DataLoader
from torch.utils.data._utils.collate import default_collate
from torch import autograd

from typing import List, Dict, Any

from transformers import Trainer as HFTrainer
from transformers import PreTrainedModel, AdamW
from transformers.file_utils import is_apex_available
from transformers.data.processors.squad import SquadResult
from transformers.data.metrics.squad_metrics import squad_evaluate, compute_predictions_logits

from kitanaqa.trainer.custom_schedulers import get_custom_exp, get_custom_linear
from kitanaqa import get_logger

# Init logging
logger = get_logger()

#autograd.set_detect_anomaly(True)

if is_apex_available():
    from apex import amp


def tensor_to_list(tensor):
    """ Convert a Tensor to List """
    return tensor.detach().cpu().tolist()


class Trainer(HFTrainer):
    """ A class to provide the adversarial and augmented training and evaluation
    ...

    Attributes
    ----------
    model_args: dataclass, 'optional' (?)
      The arguments to tweak for training. Will default to a basic instance in arguments.py if not provided. For a list of the model_args, refer to arguments.py.

    Methods
    ----------
    log(logs, iterator = None)
      Modified from HFTrainer base class, Log :obj:`logs` on the various objects watching training.

    training_step(model, batch)
      Performs one step of training (might be adversarial and/or augmented) and returns the loss.

    evaluate(prefix, args, tokenizer, dataset, examples, features)
      Performs the evaluation on the dataset and returns the evaluation metrics (Exact Match (EM) and F1-score).

    """
    def __init__(self, model_args=None, **kwargs):
        """
        Parameters
        ----------
        model_args : dataclass
            The model arguments. For a list of the model_args, check the arguments.py.
        """
        super().__init__(**kwargs)

        # Use torch default collate to bypass native
        # HFTrainer collater when using SQuAD dataset
        if not kwargs['data_collator']:
            self.data_collator = default_collate

        self.args = kwargs['args']

        # Need to get do_alum from model_args 
        self.params = model_args

        if self.params and self.params.do_alum:
            self.do_alum = True
        else:
            self.do_alum = False

        if self.do_alum and self.params.model_type not in ['bert','distilbert','albert']:
            msg = 'Only bert, albert, distilbert models are support in ALUM training'
            raise NotImplementedError(msg)

        if self.do_alum and self.args.do_train:
            # Initialize delta for ALUM adv grad
            self._delta = None
            # Set ALUM optimizer
            self._alum_optimizer = None
            # Set ALUM scheduler
            if self.params.alpha_schedule == 'exp' and self.params.alpha_final:
                self._alpha_scheduler = get_custom_exp(
                                max_steps = self.args.num_train_epochs,
                                start_val = self.params.alpha,
                                end_val = self.params.alpha_final
                            )
            elif self.params.alpha_schedule == 'linear' and self.params.alpha_final:
                self._alpha_scheduler = get_custom_linear(
                                max_steps = self.args.num_train_epochs,
                                start_val = self.params.alpha,
                                end_val = self.params.alpha_final
                            )
            else:
                self._alpha_scheduler = itertools.repeat(self.params.alpha, self.args.num_train_epochs)

            # Set static embedding layer
            if self.params.model_type == 'bert':
                self._embed_layer = self.model.bert.get_input_embeddings()
            elif self.params.model_type == 'distilbert':
                self._embed_layer = self.model.distilbert.get_input_embeddings()
            # ALUM step template
            self._step = self._alum_step
            # Tracking training steps for ALUM grad accumulation
            self._step_idx = 0
            self._n_steps = len(self.get_train_dataloader())
            self._alpha = None
        elif self.args.do_train:
            # Use non-ALUM training step
            self._step = self._normal_step

    def _normal_step(
            self,
            model: nn.Module,
            batch: List) -> torch.Tensor:

        model.train()
        batch = tuple(t.to(self.args.device) for t in batch)

        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "start_positions": batch[3],
            "end_positions": batch[4],
        }
        if self.params.model_type in ["xlm", "roberta", "distilbert"]:
            del inputs["token_type_ids"]

        outputs = model(**inputs)
        # model outputs are always tuple in transformers (see doc)
        loss = outputs[0]

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16: #  assumes using apex
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.detach()

    def setup_comet(self):
        pass

    def log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
        """
        Modified from HF Trainer base class

        Log :obj:`logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (:obj:`Dict[str, float]`):
                The values to log.
            iterator (:obj:`tqdm`, `optional`):
                A potential tqdm progress bar to write the logs on.
        """
        if self.epoch is not None:
            logs["epoch"] = self.epoch
        if self.global_step is None:
            # when logging evaluation metrics without training
            self.global_step = 0

        output = {**logs, **{"step": self.global_step}}
        if self.do_alum:
            output = {**output, **{"alpha": self._alpha}}
        if iterator is not None:
            iterator.write(output)
        else:
            print(output)

    def _alum_step(
            self,
            model: nn.Module,
            batch: List) -> torch.Tensor:
        """ Training step using ALUM """

        if (self._step_idx + 1) == self._n_steps:
            # Reset step count each epoch
            self._step_idx = 0

        if self._step_idx == 0:
            # Initialize alpha
            self._alpha = next(self._alpha_scheduler)

        batch = tuple(t.to(self.args.device) for t in batch)
        X = batch[0]  # input
        with torch.no_grad():
            input_embedding = self._embed_layer(X)

        '''
        In adversarial training, inject noise at embedding level, don't update embedding layer
        When we set input_ids = None, and inputs_embeds != None, BertEmbedding.word_embeddings won't be invoked and won't be updated by back propagation
        '''
        inputs = {
            "input_ids": None,
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "start_positions": batch[3],
            "end_positions": batch[4],
            "inputs_embeds": input_embedding,
        }
        if self.params.model_type in ["xlm", "roberta", "distilbert"]:
            del inputs["token_type_ids"]

        # Initialize delta for every actual batch
        if self._step_idx % self.args.gradient_accumulation_steps == 0:
            m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(768),torch.eye(768)*(self.params.sigma ** 2))
            # TODO: clamp distribution
            sample = m.sample((self.params.max_seq_length,))
            self._delta = torch.tensor(sample, requires_grad = True, device = self.args.device)
            if not self._alum_optimizer:
                optimizer_params = [
                    {
                        "params": self._delta
                    }
                ]
                opt = AdamW(optimizer_params)
                if self.args.fp16:
                    _, self._alum_optimizer = amp.initialize(
                                                self.model,
                                                opt,
                                                opt_level=self.args.fp16_opt_level)
                else:
                    self._alum_optimizer = opt

        # Predict logits and generate normal loss with normal inputs_embeds
        outputs = model(**inputs)
        normal_loss, start_logits, end_logits = outputs[0:3]
        start_logits, end_logits = torch.argmax(start_logits, dim=1), torch.argmax(end_logits, dim=1)

        # Generation of attack shouldn't affect the gradients of model parameters
        # Set model to inference mode and disable accumulation of gradients
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # Iterative attack
        for i in range(self.params.K):
            # Generate adversarial gradients with perturbed inputs and target = predicted logits
            inputs = {
                "input_ids": None,
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": start_logits,
                "end_positions": end_logits,
                "inputs_embeds": input_embedding + self._delta,
            }
            if self.params.model_type in ["xlm", "roberta", "distilbert"]:
                del inputs["token_type_ids"]

            outputs = model(**inputs)
            adv_loss = outputs[0]

            if self.args.n_gpu > 1:
                adv_loss = adv_loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if self.args.gradient_accumulation_steps > 1:
                adv_loss = adv_loss / self.args.gradient_accumulation_steps

            # Accumulating gradients for delta (g_adv) only, model gradients are not affected because we set model.eval()
            if self.args.fp16:
                # Gradients are unscaled during context manager exit.
                with amp.scale_loss(adv_loss, self._alum_optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                adv_loss.backward()

            # Check for inf/NaN in delta grad. These can be introduced by instability in mixed-precision training.
            if not torch.all(torch.isfinite(self._delta.grad.data)):
                logger.warning('Detected inf/NaN in adv gradient. Zeroing and continuing accumulation')
                self._alum_optimizer.zero_grad()

            # Calculate g_adv and update delta every actual epoch
            if (self._step_idx + 1) % self.args.gradient_accumulation_steps == 0:
                g_adv = self._delta.grad.data.detach()
                self._delta.data = self._alum_grad_project((self._delta + self.params.eta * g_adv), self.params.eps, 'inf')
                del g_adv

        # Set model to train mode and enable accumulation of gradients
        for param in model.parameters():
            param.requires_grad = True
        model.train()

        # Generate adversarial loss with perturbed inputs against predicted logits
        inputs = {
            "input_ids": None,
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "start_positions": start_logits,
            "end_positions": end_logits,
            "inputs_embeds": input_embedding + self._delta,
        }
        if self.params.model_type in ["xlm", "roberta", "distilbert"]:
            del inputs["token_type_ids"]

        outputs = model(**inputs)
        adv_loss = outputs[0]

        loss = normal_loss + self._alpha * adv_loss
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        # Accumulating gradients for all parameters in the model
        if self.args.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        self._step_idx += 1

        return normal_loss.detach()


    def training_step(
            self,
            model: nn.Module,
            batch: List,
        ) -> torch.Tensor:
        """Performs one step of training (might be adversarial and/or augmented)

        Parameters
        ----------
        model : nn.Module
            The model to be used for training
        batch : List
            The btach used for one step of training. Includes the input_ids, attention_masks, token_type_ids, start_positions, and end_positions

        Returns
        -------
        torch.Tensor
            The training loss after one step of training
        """
        return self._step(model, batch)

    def _alum_grad_project(self, grad, eps, ord = 'inf'):
        if ord == 2:
            dims = list(range(1, grad.dim()))
            norms = torch.sqrt(torch.sum(grad * grad, dim=dims, keepdim=True))
            return torch.min(torch.ones(norms.shape), eps / norms) * grad
        elif ord == 'inf':
            return torch.clamp(grad, min = -eps, max = eps)
        else:
            msg = "Only ord = inf and ord = 2 have been implemented"
            raise NotImplementedError(msg)

    def _adv_sgn_attack(self, delta, eps, eps_iter, ord = 'inf'):
        if ord == 'inf':
            grad_sign = delta.grad.data.sign()
            delta.data += eps * grad_sign
            delta.data = torch.clamp(delta.data, min = -eps, max = eps)
            return delta
        else:
            msg = "Only ord = inf has been implemented"
            raise NotImplementedError(msg)


    def adv_evaluate(
            self,
            prefix: str,
            args,
            tokenizer,
            dataset,
            examples,
            features) -> torch.Tensor:
        """Performs PGD attack on each example in the evaluation dataset, recording aggregate metrics

        Parameters
        ----------
        prefix : str
            The model to be used for training
        args :

        tokenizer : 
            The tokenizer used to preprocess the data.

        dataset : List(torch.utils.data.TensorDataset)
            The evaluation dataset

        examples : List(torch.utils.data.TensorDataset)
            The examples in the evaluation dataset

        features : List(torch.utils.data.TensorDataset)
            SQuAD-like features corresponding to the evalaution dataset

        Returns
        -------
        torch.Tensor
            The evaluation metrics (Exact Match (EM) and F1-score)
        """

        if not os.path.exists(self.args.output_dir) and self.args.local_rank in [-1, 0]:
            os.makedirs(self.args.output_dir)

        # TODO Add batch attacks for eval
        eval_batch_size = max(1, self.args.n_gpu)

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        # multi-gpu evaluate
        if self.args.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", eval_batch_size)

        all_results = []
        start_time = timeit.default_timer()

        if self.params.model_type == 'bert':
            _embed_layer = self.model.bert.get_input_embeddings()
        elif self.params.model_type == 'distilbert':
            _embed_layer = self.model.distilbert.get_input_embeddings()
        elif self.params.model_type == 'albert':
            _embed_layer = self.model.albert.get_input_embeddings()

        self.model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.args.device) for t in batch)
            adv_outputs = []  # (k_iter, batch_size)
            # Set static embedding layer
            _delta = None
            for i_iter in range(args.K):
                input_embedding = torch.stack([_embed_layer(x) for x in batch[0]])
                if not _delta:
                    m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(768),torch.eye(768)*(args.sigma ** 2))
                    _sample = m.sample((args.max_seq_length,))
                    _delta = torch.tensor(_sample, requires_grad = True, device = self.args.device)

                adv_input_embedding = input_embedding + _delta
                inputs = {
                    "input_ids": None,
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "start_positions": batch[3],
                    "end_positions": batch[4],
                    "inputs_embeds": adv_input_embedding,
                }

                if self.params.model_type in ["xlm", "roberta", "distilbert"]:
                    del inputs["token_type_ids"]

                intermed_adv_outputs = self.model(**inputs)

                adv_loss = intermed_adv_outputs[0]
                adv_loss.backward()

                # Calculate g_adv and update delta
                g_adv = _delta.grad.data.detach()
                _delta = self._adv_sgn_attack(_delta, args.eps, args.eta, 'inf')
                del g_adv
            _delta.grad.zero_()

            # TODO: Check inf/NaN. How should we proceed with eval if NaNs?

            with torch.no_grad():
                # Generate adversarial loss with perturbed inputs against predicted logits
                inputs = {
                    "input_ids": None,
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "inputs_embeds": input_embedding + _delta
                }

                if self.params.model_type in ["xlm", "roberta", "distilbert"]:
                    del inputs["token_type_ids"]

                adv_outputs = self.model(**inputs)

                example_indices = batch[5]

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)

                adv_output = [tensor_to_list(output[i]) for output in adv_outputs]

                start_logits, end_logits = adv_output
                result = SquadResult(unique_id, start_logits, end_logits)
                example_id = example_index.item()
                all_results.append(result)

        eval_time = timeit.default_timer() - start_time
        logger.info("  Evaluation done in total %f secs (%f sec per example)", eval_time, eval_time / len(dataset))

        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            None,
            None,
            None,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer,
        )
        alum_results = squad_evaluate(examples, predictions) 
        print('===alum_results: ', alum_results)
        return alum_results

    
    def evaluate(
            self,
            prefix: str,
            args,
            tokenizer,
            dataset,
            examples,
            features) -> torch.Tensor:
        """Performs evaluation on the dataset

        Parameters
        ----------
        prefix : str
            The model to be used for training
        args :

        tokenizer : 
            The tokenizer used to preprocess the data.

        dataset : List(torch.utils.data.TensorDataset)
            The evaluation dataset

        examples : List(torch.utils.data.TensorDataset)
            The examples in the evaluation dataset

        features : List(torch.utils.data.TensorDataset)
            SQuAD-like features corresponding to the evalaution dataset

        Returns
        -------
        torch.Tensor
            The evaluation metrics (Exact Match (EM) and F1-score)
        """
        if not os.path.exists(self.args.output_dir) and self.args.local_rank in [-1, 0]:
            os.makedirs(self.args.output_dir)

        eval_batch_size = self.args.per_device_eval_batch_size * max(1, self.args.n_gpu)

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        # multi-gpu evaluate
        if self.args.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", eval_batch_size)

        all_results = []
        start_time = timeit.default_timer()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):

            self.model.eval()
            batch = tuple(t.to(self.args.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                if self.params.model_type in ["xlm", "roberta", "distilbert"]:
                    del inputs["token_type_ids"]

                example_indices = batch[3]

                outputs = self.model(**inputs)

                example_indices = batch[5]

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [tensor_to_list(output[i]) for output in outputs]

                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

                all_results.append(result)

        eval_time = timeit.default_timer() - start_time
        logger.info("  Evaluation done in total %f secs (%f sec per example)", eval_time, eval_time / len(dataset))

        # Compute predictions
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            None,
            None,
            None,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer,
        )

        # Compute the F1 and exact scores.
        results = squad_evaluate(examples, predictions)
        return results
