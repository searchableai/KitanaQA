import pickle
import sys
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import math
from collections import Counter
from doggmentator.drop_words import dropwords
from doggmentator.term_replacement import *

data_file = pkg_resources.resource_filename(
            'doggmentator', 'support/SQuAD_v1.1_dev.pickle')
with open(data_file, 'rb') as f:
    score_dict = pickle.load(f)

class MyTensorDataset(Dataset):
    def __init__(self, raw_dataset, tokenizer, importance_score_dict, is_training=False, sample_ratio=0.5,
                 p_replace=0.1, p_dropword=0.1, p_misspelling=0.1):
        # A list of tuples of tensors
        self.dataset = []
        aug_dataset = []
        raw_dataset = [raw_dataset.__getitem__(idx) for idx in range(len(raw_dataset))]

        # Normalize probabilities of each augmentation
        probs = [p_dropword, p_replace, p_misspelling]
        probs = [p / sum(probs) for p in probs]
        augmentation_types = {
            'drop': dropwords,
            'synonym': ReplaceTerms(rep_type='synonym').replace_terms,
            'misspelling': ReplaceTerms(rep_type='misspelling').replace_terms
        }
        num_examples = len(raw_dataset)
        num_aug_examples = math.ceil(num_examples * sample_ratio)
        orig_indices = list(range(num_examples))

        # Randomly sample indices of data in original dataset with replacement
        aug_indices = np.random.choice(orig_indices, size=num_aug_examples)
        aug_freqs = Counter(aug_indices)

        # Reamining number of each agumentation types after exhausting previous example's variations
        remaining_count = {}
        for aug_type in augmentation_types.keys():
            remaining_count[aug_type] = 0

        ct = 0

        for aug_idx, count in aug_freqs.items():
            # Get frequency of each augmentation type for current example with replacement
            aug_type_sample = np.random.choice(list(augmentation_types.keys()), size=count, p=probs)
            aug_type_freq = Counter(aug_type_sample)
            for aug_type in aug_type_freq.keys():
                aug_type_freq[aug_type] += remaining_count[aug_type]

            # Get raw data from original dataset and get corresponding importance score
            raw_data = raw_dataset[aug_idx]
            if is_training:
                input_ids, attention_masks, token_type_ids, start_positions, end_positions, cls_index, p_mask, is_impossible = raw_data
            else:
                input_ids, attention_masks, token_type_ids, feature_index, cls_index, p_mask = raw_data
            seq_len = len(input_ids)  # sequence length for padding
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            sep_id = [idx for idx, token in enumerate(tokens) if token == '[SEP]']
            question = tokenizer.convert_tokens_to_string(tokens[1:sep_id[0]])
            context = tokenizer.convert_tokens_to_string(tokens[sep_id[0] + 1:sep_id[1]])
            importance_score = importance_score_dict[aug_idx]

            if ct % 1000 == 0:
                print(question)
                print(context)
                print(importance_score)
                print('+' * 60)
                print(aug_type_freq)
            
            for aug_type, aug_times in aug_type_freq.items():
                if aug_type == 'drop':
                    aug_questions = augmentation_types[aug_type](question, N = 1, K = 1)
                else:
                    aug_questions = augmentation_types[aug_type](sentence = question,
                                                             importance_scores = importance_score,
                                                             num_replacements = 1,
                                                             num_output_sents = aug_times,
                                                             sampling_strategy = 'topK',
                                                             sampling_k = 5)
                for aug_question in aug_questions:
                    print(aug_question)
                    data_dict = tokenizer.encode_plus(aug_question, context,
                                                      pad_to_max_length=True,
                                                      max_length=seq_len,
                                                      is_pretokenized=False,
                                                      return_token_type_ids=True,
                                                      return_attention_mask=True,
                                                      return_tensors='pt',
                                                      )
                    if is_training:
                        aug_dataset.append(tuple([data_dict['input_ids'][0],
                                                  data_dict['attention_mask'][0],
                                                  data_dict['token_type_ids'][0],
                                                  start_position,
                                                  end_position,
                                                  cls_index,
                                                  p_mask,
                                                  is_impossible,
                                                  ]))
                    else:
                        aug_dataset.append(tuple([data_dict['input_ids'][0],
                                                  data_dict['attention_mask'][0],
                                                  data_dict['token_type_ids'][0],
                                                  feature_index,
                                                  cls_index,
                                                  p_mask,
                                                  ]))
                if ct % 1000 == 0:
                    print(aug_question)
                remaining_count[aug_type] = aug_times - len(aug_questions)

            ct += 1
        self.dataset = raw_dataset + aug_dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)



# Load SQuAD Dataset
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor, squad_convert_examples_to_features
import tensorflow_datasets as tfds
tfds_examples = tfds.load("squad")
examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=True)
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
features, raw_dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=512,
            doc_stride=128,
            max_query_length=64,
            is_training=False,
            return_dataset="pt",
            #threads=args.threads,
)

dataset = MyTensorDataset(raw_dataset, tokenizer, score_dict)
