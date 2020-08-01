experiment = None
# edit the following code to track training using cometml
#from comet_ml import Experiment
#experiment = Experiment(api_key="", project_name="", workspace="")
#experiment.log_asset_folder('data')

import argparse
import glob
import logging
import os
import random
import timeit
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForQuestionAnswering,
    DistilBertTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

'''
ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig, XLNetConfig, XLMConfig)),
    (),
)
'''

MODEL_CLASSES = {
    "albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
    "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
}

try:
    from utils import set_seed, to_list, project, get_dataloader, set_total_steps, set_optimizer_and_scheduler, initialize_fp16m, QA_Trainer, load_and_cache_examples
except ImportError:
    from.utils import set_seed, to_list, project, get_dataloader, set_total_steps, set_optimizer_and_scheduler, initialize_fp16m, QA_Trainer, load_and_cache_examples

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def main():
    args = {
        'model_type' : 'bert', # name of model in MODEL_CLASSES, e.g. albert, bert, distilbert
        'model_name_or_path' : 'bert-base-uncased', # type of model, e.g. albert-xxlarge-v2, bert-large-uncased
	#'model_name_or_path' : '/home/ubuntu/bootcamp/finetune/SQuAD/train/outputs_alum/checkpoint-28000', # path to model dir, e.g. '/home/ubuntu/model/checkpoint-1000',
        'output_dir' : '/home/ubuntu/bootcamp/finetune/SQuAD/train/outputs_alum_aug', # e.g. '/home/ubuntu/model/outputs/',
        'cache_dir' : '/home/ubuntu/bootcamp/finetune/SQuAD/train/cache_alum_aug', # e.g., '/home/ubuntu/model/cache_dir',
        'data_dir' : '/home/ubuntu/bootcamp/finetune/SQuAD/train/data', # e.g., '/home/ubuntu/model/data',
        'train_file' : '/home/ubuntu/bootcamp/finetune/SQuAD/train/data/train-v1.1.json', # e.g., '/home/ubuntu/model/data/train-v2.0.json',
        'predict_file' : '/home/ubuntu/bootcamp/finetune/SQuAD/train/data/dev-v1.1.json', # e.g., '/home/ubuntu/model/data/dev-v2.0.json',
        'tokenizer_name' : '', # e.g. tokenizer for model_name e.g., 'albert-xxlarge-v2',
        'do_lower_case' : True,
        'overwrite_output_dir' : True,
        'config_name' : '',
        'max_seq_length' : 512,
	    'max_query_length' : 64,
	    'adam_epsilon' : 1e-8,
        'max_grad_norm' : 1.,
        'max_answer_length' : 30,
        'local_rank' : -1,
        'fp16_opt_level' : 'O1',
        'server_ip' : '',
        'server_port' : '',
        'doc_stride' : 128,
        'do_train' : True,
        'null_score_diff_threshold' : -2,
        'version_2_with_negative' : False, # Use for SQuAD v2.0 training
        'per_gpu_train_batch_size' : 2,
        'per_gpu_eval_batch_size' : 4,
        'learning_rate' : 3e-5,
        'num_train_epochs' : 5,
        'max_seq_length' : 512,
        'preprocess_input_data' : False,
        'warmup_ratio' : 0.0,
        'weight_decay' : 0.0,
        'logging_steps' : 50,
        'save_steps' : 1000,
        'evaluate_during_training' : False,
        'gradient_accumulation_steps' : 16,
        'max_steps' : -1, #-1 to default to num epochs
        'seed' : 512,
        'fp16' : True,
        #'threads' : 4,
    }

    args = dotdict(args)
    if experiment:
        experiment.log_parameters(args)

    from apex import amp

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
        trainer = QA_Trainer(args, train_dataset, model, tokenizer)
        global_step, tr_loss = trainer.train_model()
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(dict(args), os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)  # , force_download=True)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
                logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
        else:
            logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
            checkpoints = [args.model_name_or_path]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)  # , force_download=True)
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer, prefix=global_step)

            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))

    return results


if __name__ == "__main__":
    if experiment:
        with experiment.train():
    	    main()
    else:
        main()