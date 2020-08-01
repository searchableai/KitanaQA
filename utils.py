import random
import numpy as np
import torch


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()

def project(X, eps, ord = 'inf'):
    if ord == 2:
        dims = list(range(1, X.dim()))
        norms = torch.sqrt(torch.sum(X * X, dim=dims, keepdim=True))
        return torch.min(torch.ones(norms.shape), eps / norms) * X
    else:
        return torch.clamp(X, min = -eps, max = eps)

def get_dataloader(dataset, local_rank, batch_size) -> Dataloader:
    sampler = RandomSampler(dataset) if local_rank == -1 else DistributedSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader

def set_total_steps(args, train_dataloader) -> t_total:
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    return t_total

def set_optimizer_and_scheduler(args, t_total, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if not args.warmup_steps:
        args.warmup_steps = int(args.warmup_ratio * t_total)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    return optimizer, scheduler

def initialize_fp16(args, model, optimizer):
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        return model, optimizer

def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
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

        if not args.data_dir and ((evaluate and not args.predict_file) or (not evaluate and not args.train_file)):
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
                examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
            else:
                examples = processor.get_train_examples(args.data_dir, filename=args.train_file)
                aug_examples = processor.get_train_examples(args.data_dir, filename = '/home/ubuntu/bootcamp/finetune/SQuAD/train/data/checkpoint-squad.json')
                examples = examples + aug_examples
                print('Concatenete augmented examples to original examples. Total length = ', len(examples))
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

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset

class QA_Trainer():

    def __init__(self, args, train_dataset, model, tokenizer):
        self.args = args
        self.train_dataset = train_dataset
        self.model = model
        self.tokenizer = tokenizer

        if self.args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()
        self.args.train_batch_size = self.args.per_gpu_train_batch_size * max(1, self.args.n_gpu)
        self.train_dataloader = get_dataloader(self.train_dataset, self.args.local_rank, self.args.train_batch_size)

        self.t_total = set_total_steps(self.args, self.train_dataloader)

        self.optimizer, self.scheduler = set_optimizer_and_scheduler(self.args, self.t_total, model = self.model)

        self.model, self.optimizer = initialize_fp16(self.args, self.model, self.optimizer)

        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
            )

    def train_model(self):
        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", self.args.per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1),
        )
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", self.t_total)

        global_step = 1
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if os.path.exists(self.args.model_name_or_path):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = self.args.model_name_or_path.split("-")[-1].split("/")[0]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (len(self.train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (
                            len(self.train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch *= self.args.gradient_accumulation_steps

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                logger.info("  Starting fine-tuning.")

        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(self.args.num_train_epochs), desc="Epoch", disable=self.args.local_rank not in [-1, 0]
        )
        # Added here for reproductibility
        set_seed(args)

        # Adversarial Training Setting
        embed_layer = self.model.bert.get_input_embeddings()
        eta = 1e-3  # step size of PGD
        start_alpha = 10  # 1 for fine tuning, 10 for pre training
        end_alpha = 1
        alpha = 1
        eps = 1e-5  # perturbation range
        sigma = 1e-5  # variance for multivariate normal distribution, mean is 0
        K = 1
        lambd = math.log(start_alpha / end_alpha, (1 + self.t_total))

        print('lambda exponent:', lambd)

        for _ in train_iterator:
            epoch_iterator = tqdm(self.train_dataloader, desc="Iteration", disable= self.args.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                batch = tuple(t.to(args.device) for t in batch)
                X = batch[0]  # input
                with torch.no_grad():
                    input_embedding = embed_layer(X)

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

                if self.args.model_type in ["xlm", "roberta", "distilbert"]:
                    del inputs["token_type_ids"]

                if self.args.model_type in ["xlnet", "xlm"]:
                    inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
                    if self.args.version_2_with_negative:
                        inputs.update({"is_impossible": batch[7]})

                # Initialize delta for every actual batch, 768 for base and 1024 for large. Modified based on embedding layer size
                if step % self.args.gradient_accumulation_steps == 0:
                    embed_size = len(input_embedding[0][1])
                    m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(embed_size),
                                                                                   torch.eye(embed_size) * (sigma ** 2))
                    sample = m.sample((self.args.max_seq_length,))
                    delta = torch.tensor(sample, requires_grad=True, device=self.args.device)

                # Predict logits and generate normal loss with normal inputs_embeds
                outputs = model(**inputs)
                normal_loss, start_logits, end_logits = outputs[0:3]
                start_logits, end_logits = torch.argmax(start_logits, dim=1), torch.argmax(end_logits, dim=1)

                # Generation of attack shouldn't affect the gradients of model parameters
                # Set model to inference mode and disable accumulation of gradients
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False
                # Iterative attack
                for i in range(K):
                    # Generate adversarial gradients with perturbed inputs and target = predicted logits
                    inputs = {
                        "input_ids": None,
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2],
                        "start_positions": start_logits,
                        "end_positions": end_logits,
                        "inputs_embeds": input_embedding + delta,
                    }
                    outputs = model(**inputs)
                    adv_loss = outputs[0]

                    if self.args.n_gpu > 1:
                        adv_loss = adv_loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
                    if self.args.gradient_accumulation_steps > 1:
                        adv_loss = adv_loss / args.gradient_accumulation_steps

                    # Accumulating gradients for delta (g_adv) only, model gradients are not affected because we set model.eval()
                    if self.args.fp16:
                        with amp.scale_loss(adv_loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        adv_loss.backward()

                    # Calculate g_adv and update delta every actual epoch
                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        # print(delta)
                        # print(delta.grad)
                        g_adv = delta.grad.data
                        delta.data = project((delta + eta * g_adv), eps, 'inf')
                        delta.grad.zero_()
                        del g_adv
                # Set model to train mode and enable accumulation of gradients
                for param in self.model.parameters():
                    param.requires_grad = True
                self.model.train()
                # Generate adversarial loss with perturbed inputs against predicted logits
                inputs = {
                    "input_ids": None,
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "start_positions": start_logits,
                    "end_positions": end_logits,
                    "inputs_embeds": input_embedding + delta,
                }
                outputs = model(**inputs)
                adv_loss = outputs[0]

                loss = normal_loss + alpha * adv_loss
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

                tr_loss += normal_loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()  # Clear gradients for next batch
                    global_step += 1
                    # alpha = start_alpha / (1 + global_step) ** lambd

                    # Log metrics
                    if self.args.local_rank in [-1,
                                           0] and self.args.logging_steps > 0 and global_step % self.args.logging_steps < self.args.gradient_accumulation_steps:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        if self.args.local_rank == -1 and self.args.evaluate_during_training:
                            results = evaluate(self.args, self.model, self.tokenizer)
                            for key, value in results.items():
                                tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss) / self.args.logging_steps, global_step)
                        logging_loss = tr_loss

                    # Save model checkpoint
                    if self.args.local_rank in [-1, 0] and self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        logger.info("SAVING: %s step, %s mod grad_acc, %s global_step, %s mod save_steps --", str(step),
                                    str((step + 1) % self.args.gradient_accumulation_steps), str(global_step),
                                    str(global_step % self.args.save_steps))
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # Take care of distributed/parallel training
                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(dict(self.args), os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(self.scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

                if self.args.max_steps > 0 and global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break
            if self.args.max_steps > 0 and global_step > self.args.max_steps:
                train_iterator.close()
                break

        if self.args.local_rank in [-1, 0]:
            tb_writer.close()

        return global_step, tr_loss / global_step



    def eval_model(self, prefix = ""):
        dataset, examples, features = load_and_cache_examples(self.args, self.tokenizer, evaluate=True, output_examples=True)

        if not os.path.exists(self.args.output_dir) and self.args.local_rank in [-1, 0]:
            os.makedirs(self.args.output_dir)

        self.args.eval_batch_size = self.args.per_gpu_eval_batch_size * max(1, self.args.n_gpu)

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)


        # multi-gpu evaluate
        if self.args.n_gpu > 1 and not isinstance(self.model, torch.nn.DataParallel):
            self.model = torch.nn.DataParallel(self.model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)

        all_results = []
        start_time = timeit.default_timer()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                if self.args.model_type in ["xlm", "roberta", "distilbert"]:
                    del inputs["token_type_ids"]

                example_indices = batch[3]

                # XLNet and XLM use more arguments for their predictions
                if self.args.model_type in ["xlnet", "xlm"]:
                    inputs.update({"cls_index": batch[4], "p_mask": batch[5]})

                outputs = self.model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [to_list(output[i]) for output in outputs]

                # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
                # models only use two.
                if len(output) >= 5:
                    start_logits = output[0]
                    start_top_index = output[1]
                    end_logits = output[2]
                    end_top_index = output[3]
                    cls_logits = output[4]

                    result = SquadResult(
                        unique_id,
                        start_logits,
                        end_logits,
                        start_top_index=start_top_index,
                        end_top_index=end_top_index,
                        cls_logits=cls_logits,
                    )

                else:
                    start_logits, end_logits = output
                    result = SquadResult(unique_id, start_logits, end_logits)

                all_results.append(result)

        evalTime = timeit.default_timer() - start_time
        logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

        # Compute predictions
        output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
        output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))

        if args.version_2_with_negative:
            output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
        else:
            output_null_log_odds_file = None

        # XLNet and XLM use a more complex post-processing procedure
        if args.model_type in ["xlnet", "xlm"]:
            start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
            end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

            predictions = compute_predictions_log_probs(
                examples,
                features,
                all_results,
                args.n_best_size,
                args.max_answer_length,
                output_prediction_file,
                output_nbest_file,
                output_null_log_odds_file,
                start_n_top,
                end_n_top,
                args.version_2_with_negative,
                tokenizer,
                args.verbose_logging,
            )
        else:
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