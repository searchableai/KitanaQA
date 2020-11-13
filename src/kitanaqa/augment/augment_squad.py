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
from kitanaqa.augment.term_replacement import *
from kitanaqa import get_logger


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
                raw_examples: Dict,
                custom_importance_scores: Dict=None,
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
                verbose: bool=False):
        """ Dataset class to generate perturbations of SQuAD-like data
        ...
        Methods
        ----------
        generate()
          Generate perturbations for the input dataset using init params.

        Parameters
        ----------
        raw_examples : Dict
            Original examples to perturb. These should minimally match the SQuAD data format. Additional field are ignored.
        custom_importance_scores : Optional(Dict)
            Dictionary with keys matching each question ID (qid) value in the original raw_examples. Values are List of Tuples containing (term, weight) pairs for the tokenized input questions.
        is_training : Optional(bool)
            Flag defining whether perturbed data will be used for evaluation or training. The default is False. If True, each example of generated SQuAD-like data is also annotated with the `is_impossible` field from the original question (qid).
        sample_ratio : Optional(float)
            Target number of examples to generate for each example in the original raw_examples. For example, given value of 1.0, would try to generate one perturbed example for each original example, resulting in a 1:1 ratio of original to perturbed. The default value is 4.
        num_replacements : Optional(int)
            Target number of terms to replace in the original sentence when using replacement perturbations. The default value is 2.
        sampling_k : Optional(int)
            The number of terms in the importance score vector to include in topK or bottomK sampling. This parameter is not used by the default sampling_strategy, `random` sampling.
        sampling_strategy : Optional(str)
            Strategy used to sample terms to perturb in the original sentence. The default is random. If custom_importance_scores is given, then sampling_strategy may be `topK` or `bottomK`, in which case the custom_importance_scores (or inverted scores) vector is used for weighted sampling.
        p_replace : Optional(float)
            Sampling probability for the replacement perturbation. If given, will be normalized with all other perturbation sampling probabilities. The default is uniform sampling across all perturbations.
        p_dropword : Optional(float)
            Sampling probability for the replacement perturbation. If given, will be normalized with all other perturbation sampling probabilities. The default is uniform sampling across all perturbations.
        p_misspelling : Optional(float)
            Sampling probability for the misspelling perturbation. If given, will be normalized with all other perturbation sampling probabilities. The default is uniform sampling across all perturbations.
        save_freq : Optional(int)
            A checkpoint file will be saved every save_freq examples in the raw_examples data. The default value is 100.
        from_checkpoint : Optional(bool)
            Flag to read augmented data from a previous saved state. We will check for a checkpoint.pkl file in the working directory and continue augmentation on the raw_examples data. The default value is False.
        out_prefix : Optional(str)
            Tag used to denote the saved results files if verbose is True. The default is None. If not specified, will be set to either `train` if is_training=True, or `dev` otherwise.
        verbose : Optional(bool)
            Flag to enable verbose logging
        """

        if is_training and not out_prefix:
            self.out_prefix = 'train'
        elif out_prefix:
            self.out_prefix = out_prefix
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
        self.from_checkpoint = from_checkpoint
        self.save_freq = save_freq
        self.custom_importance_scores = custom_importance_scores
        logger.info('Running SQuADDataset with hparams {}'.format(self.hparams))

        self.aug_dataset = []
        self.dataset = []
        self.formatted_dataset = {}
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
        """ Generate perturbations for the raw SQuAD-like examples
        Parameters
        ----------
        term : str
            The input term for which we are looking for synonyms.
        num_syns : Optional(int)
            The number of synonyms for the input term. The number of synonyms should be greater than 1. The default value is 10.
        similarity_thre : Optional(float)
            The similarity threshold. The function returns the synonyms with higher similarity than the threshold.

        Returns
        -------
        None

        Example
        -------
        >>> from augment_squad import SQuADDataset
        >>> with open('support/squad-dev-v1.1.json', 'r') as f:
        >>>     squad_dev_examples = json.read(f)
        >>> ds = SQuADDataset(squad_dev_examples, sample_ratio = 0.0001)
        >>> ds.generate()
        >>> ds()
        """
        aug_examples = copy.deepcopy(self.examples)

        # Randomly sample indices of data in original dataset with replacement
        aug_indices = np.random.choice(self.orig_indices, size=self.num_aug_examples)
        aug_freqs = Counter(aug_indices)

        ct = 0
        if self.from_checkpoint:
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
            if len(self.aug_dataset) >= self.num_aug_examples:
                break
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
                
            if self.custom_importance_scores and qid in self.custom_importance_scores:
                importance_score = self.custom_importance_scores[qid]
            else:
                importance_score = None

            if ct % self.save_freq == 0 and ct > 0:
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
                reps = np.random.choice(np.arange(self.hparams['num_replacements']), 1, replace=False)[0]

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
                                                            sampling_strategy = self.hparams['sampling_strategy'],
                                                            sampling_k = self.hparams['sampling_k'])
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

        self.aug_dataset = self.aug_dataset[:self.num_aug_examples]
        self.formatted_dataset = format_squad(self.aug_dataset, self.title_map, self.context_map)
        self.dataset = self.aug_dataset
        if self.verbose:
            logger.info('Saving data')
            aug_seqs = aug_seqs[:self.num_aug_examples]
            # Log the original question alongside augmented with type annotation
            with open(self.out_prefix+'_aug_seqs.json', 'w') as f:
                json.dump(aug_seqs, f)
            with open(self.out_prefix+'_aug_squad_v1.json', 'w') as f:
                json.dump(self.formatted_aug_dataset, f)
            with open('hparams.json', 'w') as f:
                json.dump(self.hparams, f)

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

    def __call__(self):
        if self.formatted_dataset:
            return self.formatted_dataset
        else:
            return None


if __name__ == "__main__":
    pass
