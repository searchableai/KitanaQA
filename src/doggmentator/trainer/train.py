import torch
import os
import logging
import timeit
from tqdm import tqdm

from torch import nn
from torch.utils.data import SequentialSampler, DataLoader
from torch.utils.data._utils.collate import default_collate
from typing import List, Dict, Any

from transformers import Trainer as HFTrainer
from transformers.file_utils import is_apex_available

from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate
)

from transformers.data.processors.squad import SquadResult

logger = logging.getLogger(__name__)

if is_apex_available():
    from apex import amp


def tensor_to_list(tensor):
    """ Convert a Tensor to List """
    return tensor.detach().cpu().tolist()


def project(X, eps, order = 'inf'):
    if order == 2:
        dims = list(range(1, X.dim()))
        norms = torch.sqrt(torch.sum(X * X, dim=dims, keepdim=True))
        return torch.min(torch.ones(norms.shape), eps / norms) * X
    else:
        return torch.clamp(X, min = -eps, max = eps)


class Trainer(HFTrainer):
    def __init__(self, model_args=None, **kwargs):
        super().__init__(**kwargs)

        # Use torch default collate to bypass native
        # HFTrainer collater when using SQuAD dataset
        if not kwargs['data_collator']:
            self.data_collator = default_collate

        self.args = kwargs['args']

        # Need to get do_alum from model_args 
        self.params = model_args
        if self.params and self.params.do_alum:
            # Set static embedding layer
            self._embed_layer = self.model.bert.get_input_embeddings()
            # ALUM step template
            self._step = self._alum_step
            # Tracking training steps for ALUM grad accumulation
            self._step_idx = 0
            self._n_steps = len(self.get_train_dataloader())
            # Initialize delta for ALUM adv grad
            self._delta = None
        else:
            # Use non-ALUM training step
            self._step = self._normal_step

    def _normal_step(
            self,
            model: nn.Module,
            batch: List,
            optimizer: torch.optim.Optimizer) -> float:

        model.train()
        batch = tuple(t.to(self.args.device) for t in batch)

        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "start_positions": batch[3],
            "end_positions": batch[4],
        }

        outputs = model(**inputs)
        # model outputs are always tuple in transformers (see doc)
        loss = outputs[0]

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16: #  assumes using apex
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.item()

    def _alum_step(
            self,
            model: nn.Module,
            batch: List,
            optimizer: torch.optim.Optimizer) -> float:

            if self._n_steps < self._step_idx:
                self._step_idx = 0
            logger.debug('_step {} - of - {}'.format(self._step_idx, self._n_steps))

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

            # Initialize delta for every actual batch
            if self._step_idx % self.args.gradient_accumulation_steps == 0:
                m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(768),torch.eye(768)*(self.params.sigma ** 2))
                sample = m.sample((self.params.max_seq_length,))
                self._delta = torch.tensor(sample, requires_grad = True, device = self.args.device)
                # delta = torch.nn.Parameter(delta)
                # delta.requires_grad = True
                # delta = delta.to(self.args.device)

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
                outputs = model(**inputs)
                adv_loss = outputs[0]

                if self.args.n_gpu > 1:
                    adv_loss = adv_loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
                if self.args.gradient_accumulation_steps > 1:
                    adv_loss = adv_loss / self.args.gradient_accumulation_steps

                # Accumulating gradients for delta (g_adv) only, model gradients are not affected because we set model.eval()
                if self.args.fp16:
                    with amp.scale_loss(adv_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    adv_loss.backward()

                # Calculate g_adv and update delta every actual epoch
                # TODO: confirm this update should occur only once per mini-batch
                if (self._step_idx + 1) % self.args.gradient_accumulation_steps == 0:
                    # print(delta)
                    # print(delta.grad)
                    g_adv = self._delta.grad.data
                    self._delta.data = project((self._delta + self.params.eta * g_adv), self.params.eps, 'inf')
                    self._delta.grad.zero_()
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
            outputs = model(**inputs)
            adv_loss = outputs[0]

            loss = normal_loss + self.params.alpha * adv_loss
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            # Accumulating gradients for all parameters in the model
            if self.args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            self._step_idx += 1

            return loss.item()


    def _training_step(
            self,
            model: nn.Module,
            batch: List,
            optimizer: torch.optim.Optimizer
        ) -> float:
        return self._step(model, batch, optimizer)
    
    def evaluate(
            self,
            prefix: str,
            args,
            tokenizer,
            dataset,
            examples,
            features):
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

                example_indices = batch[3]

                outputs = self.model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [tensor_to_list(output[i]) for output in outputs]

                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

                all_results.append(result)

        evalTime = timeit.default_timer() - start_time
        logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

        # Compute predictions
        '''
        output_prediction_file = os.path.join(self.args.output_dir, "predictions_{}.json".format(prefix))
        output_nbest_file = os.path.join(self.args.output_dir, "nbest_predictions_{}.json".format(prefix))

        if args.version_2_with_negative:
            output_null_log_odds_file = os.path.join(self.args.output_dir, "null_odds_{}.json".format(prefix))
        else:
            output_null_log_odds_file = None
        '''
        output_null_log_odds_file = output_nbest_file = output_prediction_file = None

        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer,
        )

        # Compute the F1 and exact scores.
        results = squad_evaluate(examples, predictions)
        return results

