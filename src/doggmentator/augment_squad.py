import pickle
import hashlib
import sys
from torch.utils.data import Dataset
import torch
import copy
from torch.utils.data import DataLoader
import math
from collections import Counter
from datetime import datetime
from doggmentator.term_replacement import *
from doggmentator import get_logger


logger = get_logger()


def _from_checkpoint(
        fname: str='checkpoint.pkl') -> Dict:
    """ Load a checkpoint file """
    with open(fname, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint

def format_squad(
        examples: Dict,
        title_map: Dict,
        context_map: Dict,
        version: str='1.1') -> Dict:
    """ Convert a flat list of dicts to nested SQuAD format """
    squad_data = {}
    squad_data['version'] = version
    squad_data['data'] = []
    qas = []
    current_tle_id = None
    ctx_ids = []
    tle_ids = []
    graphs = []
    num_examples = len(examples)
    unique_ids = [str(x) for x in range(num_examples)]

    dataset = {}
    for example in examples:
        qid = example['id']
        ctx_id = example['ctx_id']
        tle_id = example['tle_id']
        if tle_id not in dataset:
            dataset[tle_id] = {}
        if ctx_id not in dataset[tle_id]:
            dataset[tle_id][ctx_id] = []
        if not all([x['text'] for x in example['answers']]):
            raise Exception('No answer found')
        if not all([x['answer_start'] is not None for x in example['answers']]):
            raise Exception('No answer_start found')
        if not example['question']:
            logger.info('No question found: {}'.format(example['aug_type']))
            continue
        dataset[tle_id][ctx_id].append({
            'answers':example['answers'],
            'question':example['question'],
            'orig_id':qid,
            'title_id':tle_id,
            'context_id':ctx_id,
            'id':qid+unique_ids.pop(),
            'aug_type':example['aug_type']
        })

    formatted = {'version':version,'data':[]}
    for k,v in dataset.items():
        graphs = []
        for j,q in v.items(): 
            graphs.append({
                'context':context_map[j],
                'qas':q
            })
        formatted['data'].append({
            'title':title_map[k],
            'paragraphs':graphs
        })
    return formatted
            

class SQuADDataset(Dataset):
    def __init__(
                self,
                raw_examples: List,
                importance_score_dict: List[tuple]=None,
                is_training: bool=False,
                sample_ratio: float=4.,
                num_replacements: int=2,
                sampling_k: int=3,
                sampling_strategy: str='topK',
                p_replace: float=0.1,
                p_dropword: float=0.1,
                p_misspelling: float=0.1,
                save_freq: int=100,
                from_checkpoint: bool=False,
                out_prefix: str=None,
                verbose: bool=False,
                dataset: Dict = None):
        """ Instantiate a SQuADDataset instance"""

        if is_training and not out_prefix:
            self.out_prefix = 'train'
        else:
            self.out_prefix = 'dev'
        self.is_training = is_training
        self.verbose = verbose

        self.hparams = {
            "num_replacements":num_replacements,
            "sample_ratio":sample_ratio,
            "p_replace":p_replace,
            "p_dropword":p_dropword,
            "p_misspelling":p_misspelling,
            "sampling_strategy":sampling_strategy,
            "sampling_k":sampling_k
        }
        logger.info('Running SQuADDataset with hparams {}'.format(self.hparams))

        self.aug_dataset = []
        self.dataset = []
        self.examples = []
        self.title_map = {}
        self.context_map = {}

        self._raw_examples = raw_examples
        self._load_raw_examples()

        # Normalize probabilities of each augmentation
        probs = [p_dropword, p_replace, p_misspelling]
        self.probs = [p / sum(probs) for p in probs]
        self.augmentation_types = {
            'drop': DropTerms(),
            'synonym': ReplaceTerms(rep_type='synonym'),
            'misspelling': ReplaceTerms(rep_type='misspelling')
        }
        num_examples = len(self.examples)
        self.num_aug_examples = math.ceil(num_examples * sample_ratio)
        logger.info('Generating {} aug examples from {} orig examples'.format(self.num_aug_examples, num_examples))
        self.orig_indices = list(range(num_examples))


    def _load_raw_examples(self):
        """Load a SQuAD-like dataset and add annotations for augmentation"""
        ctx_id = 0
        new_squad_examples = copy.deepcopy(self._raw_examples)
        for j,psg in enumerate(self._raw_examples['data']):
            title = psg['title']
            tle_id = j
            self.title_map[tle_id] = title
            new_squad_examples['data'][j]['title_id'] = str(tle_id)
            for n,para in enumerate(psg['paragraphs']):
                context = para['context']
                self.context_map[ctx_id] = context
                new_squad_examples['data'][j]['paragraphs'][n]['context_id'] = str(ctx_id)
                for qa in para['qas']:
                    if self.is_training:
                        self.examples.append({
                            'qid':qa['id'],
                            'ctx_id':ctx_id,
                            'tle_id':tle_id,
                            'answers':qa['answers'],
                            'question':qa['question'],
                            'is_impossible':qa.get('is_impossible', False),
                        })
                    else:
                        self.examples.append({
                            'qid':qa['id'],
                            'ctx_id':ctx_id,
                            'tle_id':tle_id,
                            'answers':qa['answers'],
                            'question':qa['question']
                        })
                ctx_id += 1

        if self.verbose:
            with open('annotated-train-squadv1.json', 'w') as f:
                json.dump(new_squad_examples, f)

    def generate(self):
        """ Generate augmented examples based on raw SQuAD-like dataset"""
        aug_examples = copy.deepcopy(self.examples)

        # Randomly sample indices of data in original dataset with replacement
        aug_indices = np.random.choice(self.orig_indices, size=self.num_aug_examples)
        aug_freqs = Counter(aug_indices)

        ct = 0
        if from_checkpoint:
            checkpoint = _from_checkpoint()
            if not checkpoint:
                raise RuntimeError('Failed to load checkpoint file')
            aug_freqs = checkpoint['aug_freqs']
            self.aug_dataset = checkpoint['aug_dataset']
            self.hparams = checkpoint['hparams']
            ct = checkpoint['ct']

        # Reamining number of each agumentation types after exhausting previous example's variations
        remaining_count = {}
        for aug_type in self.augmentation_types.keys():
            remaining_count[aug_type] = 0

        aug_seqs = []
        for aug_idx, count in aug_freqs.items():
            if len(self.aug_dataset) > self.num_aug_examples:
                continue
            # Get frequency of each augmentation type for current example with replacement
            aug_type_sample = np.random.choice(list(self.augmentation_types.keys()), size=count, p=self.probs)
            aug_type_freq = Counter(aug_type_sample)
            for aug_type in aug_type_freq.keys():
                aug_type_freq[aug_type] += remaining_count[aug_type]

            # Get raw data from original dataset and get corresponding importance score
            raw_data = self.examples[aug_idx]
            question = raw_data['question']
            answers = raw_data['answers']
            qid = raw_data['qid']
            ctx_id = raw_data['ctx_id']
            tle_id = raw_data['tle_id']
            # Used for SQuAD v2.0; not present in v1.1
            is_impossible = raw_data.get('is_impossible', False)
                
            if importance_score_dict and qid in self.importance_score_dict:
                importance_score = importance_score_dict[qid]
            else:
                importance_score = None

            if ct % save_freq == 0:
                logger.info('Generated {} examples'.format(len(self.aug_dataset)))
                checkpoint = {
                    'aug_freqs':aug_freqs,
                    'aug_dataset':self.aug_dataset,
                    'hparams':self.hparams,
                    'ct':ct
                }
                with open('checkpoint.pkl', 'wb') as f:
                    pickle.dump(checkpoint, f) 
            sys.stdout.flush()

            for aug_type, aug_times in aug_type_freq.items():
                # Randomly select a number of terms to replace
                # up to the max `num_replacements`
                reps = np.random.choice(np.arange(self.num_replacements), 1, replace=False)[0]

                if aug_type == 'drop':
                    # Generate a dropword perturbation
                    aug_questions = self.augmentation_types[aug_type].drop_terms(
                                                            question,
                                                            num_terms=reps,
                                                            num_output_sents=aug_times)
                else:
                    # Generate synonym and misspelling perturbations
                    aug_questions = self.augmentation_types[aug_type].replace_terms(
                                                            sentence = question,
                                                            importance_scores = importance_score,
                                                            num_replacements = reps,
                                                            num_output_sents = aug_times,
                                                            sampling_strategy = self.sampling_strategy,
                                                            sampling_k = self.sampling_k)
                    # Add an additional drop perturbation to each generated question
                    aug_questions += [
                                        self.augmentation_types['drop'].drop_terms(
                                                        x,
                                                        num_terms=reps,
                                                        num_output_sents=1)
                                        for x in aug_questions
                                    ]
                                                        
                for aug_question in aug_questions:
                    if self.is_training:
                        self.aug_dataset.append({
                                                'id':qid,
                                                'ctx_id':ctx_id,
                                                'tle_id':tle_id,
                                                'aug_type':aug_type,
                                                'question':aug_question,
                                                'answers':answers,
                                                'is_impossible':is_impossible
                                        })
                    else:
                        aug_seqs.append({'orig': question, 'aug': aug_question, 'type':aug_type})
                        self.aug_dataset.append({
                                                'id':qid,
                                                'ctx_id':ctx_id,
                                                'tle_id':tle_id,
                                                'aug_type':aug_type,
                                                'question':aug_question,
                                                'answers':answers,
                                        })  

                remaining_count[aug_type] = aug_times - len(aug_questions)

            ct += 1

        formatted_aug_dataset = format_squad(self.aug_dataset, self.title_map, self.context_map)
        logger.info('Saving data')
        if self.verbose:
            # Log the original question alongside augmented with type annotation
            with open(out_prefix+'_aug_seqs.json', 'w') as f:
                json.dump(aug_seqs, f)
        with open(out_prefix+'_aug_squad_v1.json', 'w') as f:
            json.dump(formatted_aug_dataset, f)
        with open('hparams.json', 'w') as f:
            json.dump(self.hparams, f)

        self.dataset = aug_dataset

    def __getitem__(self, index):
        if self.dataset:
            return self.dataset[index]
        else:
            raise Exception('Please first generate an augmented dataset')

    def __len__(self):
        if self.dataset:
            return len(self.dataset)
        else:
            raise Exception('Please first generate an augmented dataset')


if __name__ == "__main__":
    pass
