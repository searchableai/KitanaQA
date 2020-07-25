import pkg_resources
import sparknlp
import re
import random
import itertools
import re
import json
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from stop_words import get_stop_words
from typing import List, Dict, Tuple
from numpy import dot
from numpy.linalg import norm
from sparknlp.pretrained import PretrainedPipeline
from sparknlp.annotator import *
from sparknlp.common import RegexRule
from sparknlp.base import *
from doggmentator.nlp_utils.firstnames import firstnames
from doggmentator import get_logger
from doggmentator.generators import SynonymReplace, MisspReplace

nltk.download('stopwords')
from nltk.corpus import stopwords, wordnet

# init logging
logger = get_logger()

# stopwords and common names lists
stop_words = list(get_stop_words('en'))  # have around 900 stopwords
nltk_words = list(stopwords.words('english'))  # have around 150 stopwords
stop_words.extend(nltk_words)
stop_words = list(set(stop_words))
remove_list = firstnames+stop_words
remove_list = [x.lower() for x in remove_list]


def validate_inputs(
        num_replacements: int,
        num_output_sents: int,
        sampling_strategy: str=None) -> List:
    """ Ensures valid input parameters """ 

    # Ensure num inputs between 1 and 10
    num_replacements = max(num_replacements, 1)
    num_replacements = min(num_replacements, 10)
    num_output_sents = max(num_output_sents, 1)
    num_output_sents = min(num_output_sents, 10)

    if sampling_strategy not in ['topK', 'bottomK', 'random']:
        sampling_strategy = 'random'
    return [
            num_replacements,
            num_output_sents,
            sampling_strategy
        ]


def get_scores(
        tokens: List[str],
        mode: str='random',
        mode_k: int=None,
        scores: List[Tuple]=None) -> List[Tuple]:
    """ Initialize and sanitize importance scores """

    # Check mode parameters
    if scores and mode not in ['topK', 'bottomK']:
        mode = 'topK'
    if mode in ['topK', 'bottomK']:
        if not mode_k:
            mode_k = 10

    if not scores:
        # Uniform initialization
        scores = [
            1.
            if x not in remove_list
            else 0
            for x in tokens
        ]
        
        if scores and sum(scores) != 0.:
            # Normalize
            scores = [
                x/sum(scores)
                for x in scores
            ]
        scores = list(zip(tokens, scores))
    else:   
        if len(scores) != len(tokens):
            score_idx, tokens_idx = 0, 0
            final_scores = []
            while tokens_idx < len(tokens):
                if score_idx < len(scores) and scores[score_idx][0] == tokens[tokens_idx]:
                    final_scores.append(scores[score_idx])
                    score_idx += 1
                    tokens_idx += 1
                else:
                # not a whole-word token, won't find a replacement for this token, assign a 0 to this token
                    final_scores.append((tokens[tokens_idx], 0))
                    tokens_idx += 1
            scores = final_scores
            

        # Ensure score types, norm, sgn
        tokens = [x[0] for x in scores]
        scores = [abs(float(x[1])) for x in scores]

        # Invert scores if sampling least important terms
        if mode == 'bottomK':
            scores = [
                1/x if x>0. else 0.
                for x in scores
            ]

        # Select topK elements
        select_args = sorted(
                            range(
                                len(scores)
                            ),
                            key=scores.__getitem__,
                            reverse=True,
                        )[:mode_k]
        scores = [
            x
            if i in select_args
            else 0.
            for i,x in enumerate(scores)
        ]

        if scores and sum(scores) > 0.:
            # Normalize
            scores = [
                x/sum(scores)
                for x in scores
            ]
        scores = list(zip(tokens, scores))
    return scores  # [(tok, score),...]


class DropTerms():
    """ Remove terms from a sequence """
    def __init__(self):
        """Instantiate a DropTerms object"""
        pass

    def drop_terms(
            self,
            sentence: str,
            num_terms: int=1,
            num_output_sents: int=1) -> List:
        """Generate list of strings by randomly dropping terms"""

        inputs = validate_inputs(
            num_terms,
            num_output_sents)
        num_terms = inputs.pop(0)
        num_output_sents = inputs.pop(0)

        # Whitespace tokenizer
        word_tokens = word_tokenize(sentence)

        # Create list of candidate drop words
        drop_word_indices = []
        for idx, word in enumerate(word_tokens):
            if word in remove_list:
                drop_word_indices.append(idx)

        # Ensure num_terms does not exceed possible terms
        if num_terms > len(drop_word_indices):
            num_terms = len(drop_word_indices)

        new_sentences = []
        if len(drop_word_indices) == 0:
            return new_sentences

        # Randomly choose num_terms indices from all drop_word_indices
        comb = []
        if num_terms == -1:
            # Generate all possible combinations for debugging
            for r in range(1, len(drop_word_indices) + 1):
                comb += list(itertools.combinations(drop_word_indices, r))
        else:
            # Generate all combinations of num_terms stopwords
            comb = list(itertools.combinations(drop_word_indices, num_terms))

        # Randomly sample drop term combinations
        n_chosen_indices = [comb[idx] for idx in np.random.choice(len(comb), size=min(num_output_sents, len(comb)), replace=False)]
        for chosen_indices in n_chosen_indices:
            new_words = [word_tokens[idx] for idx in range(len(word_tokens)) if idx not in chosen_indices]
            new_sentence = ' '.join(new_words)
            # Check if generated sent is empty string
            if new_sentence:
                new_sentences.append(new_sentence)

        # Shuffle permutations, sanitize and slice
        new_sentences = list(set(new_sentences))
        new_sentences = [
            re.sub(r'([A-Za-z0-9])(\s+)([^A-Za-z0-9])', r'\1\3',
                    x.replace('\' s ','\'s ')
            )
            for x in new_sentences
        ]

        if len(new_sentences) < num_output_sents:
            logger.debug(
                '{}:drop_terms: unable to generate num_output_sents - {} of ({})'.format(
                    __file__.split('/')[-1],
                    len(new_sentences),
                    num_output_sents
                )
            )
        return new_sentences


class ReplaceTerms():
    """ Generate sentence variants using term replacement strategies """
    def __init__(
            self,
            rep_type: str='synonym',
            use_ner: bool=True):
        """Instantiate a ReplaceTerms object"""
        if rep_type not in ['synonym', 'misspelling']:
            logger.error(
                '{}:ReplaceTerms __init__ invalid rep_type'.format(
                    __file__.split('/')[-1]
                )
            )
            raise ValueError('Not an accepted generator type')
        self._generator = self._get_generator(rep_type)
        if not self._generator:
            raise RuntimeError('Unable to init generator')
        if use_ner:
            try:
                spark = sparknlp.start()
                self._ner_pipeline = PretrainedPipeline('recognize_entities_dl', lang='en')
            except Exception as e:
                logger.error(
                    '{}:ReplaceTerms __init__ invalid rep_type'.format(
                        __file__.split('/')[-1]
                    )
                )
                raise RuntimeError('Unable to load ner pkg')

    def _get_entities(self, sentence: str) -> Dict:
        """ Tokenize and annotate sentence """

        # Start Spark session with Spark NLP
        allowed_tags = ['PER','LOC','ORG','MISC']

        # Annotate your testing dataset
        result = self._ner_pipeline.annotate(sentence)
        toks = result['token']
        mask = [
            1 if (
                any([y in x for y in allowed_tags])
                or not toks[i].isalnum()
            )
            else 0
            for i,x in enumerate(result['ner'])
        ]
        return mask, toks


    def _get_generator(self, name: str=None):
        if name == 'synonym':
            try:
                _syn = SynonymReplace()
                return _syn
            except Exception as e:
                logger.error(
                    '{}:replace_terms: unable to load word vectors'.format(
                        __file__.split('/')[-1]
                    )
                )
                return None
            return     
        elif name == 'misspelling':
            try:
                _missp = MisspReplace()
                return _missp
            except Exception as e:
                logger.error(
                    '{}:replace_terms: unable to load misspellings'.format(
                        __file__.split('/')[-1]
                    )
                )
                return None

    def replace_terms(
            self,
            sentence: str,
            importance_scores: List=None,
            num_replacements: int=1,
            num_output_sents: int=1,
            sampling_strategy: str='random',
            sampling_k: int=None) -> List:
        """Generate list of strings by replacing specified number of terms with their synonyms"""

        inputs = validate_inputs(
            num_replacements,
            num_output_sents,
            sampling_strategy)
        num_replacements = inputs.pop(0)
        num_output_sents = inputs.pop(0)
        sampling_strategy = inputs.pop(0)

        # Extract entities in the input sentence to mask
        masked_vector, tokens = self._get_entities(sentence)

        # Check if there are enough candidate terms
        if num_replacements > (len(masked_vector) - sum(masked_vector)):
            logger.warning(
                '{}:replace_terms: unable to generate num_replacements - {} of ({})'.format(
                    __file__.split('/')[-1],
                        num_replacements,
                        len(masked_vector) - sum(masked_vector)
                    )
                )
            num_replacements = len(masked_vector) - sum(masked_vector)

        # Initialize sampling scores
        importance_scores = get_scores(
            tokens,
            sampling_strategy,
            sampling_k,
            importance_scores)

        if not importance_scores:
            return []

        # Add index and mask to importance scores
        term_score_index = [
            (word[0], i, masked_vector[i])
            for i,word in enumerate(importance_scores)
        ]

        # Store only scores for later sampling
        importance_scores = [
            x[1] if not masked_vector[i] else 0  # set masked scores to zero
            for i,x in enumerate(importance_scores)
        ]

        # Candidate terms for synonym replacement
        rep_term_indices = [
            w[1] for w in term_score_index if not w[2]
        ]

        # Create List of Lists of term variants
        generated = {
            x[0]:self._generator.generate(x[0])
            for i,x in enumerate(term_score_index)
        }

        term_variants = {
            x[0]:generated.get(x[0])
            if (i in rep_term_indices
                and not masked_vector[i]
                and generated.get(x[0]))
            else []
            for i,x in enumerate(term_score_index)
        }

        # Set scores to zero for all terms w/o synonyms
        importance_scores = [
            x
            if (
                term_score_index[i][0] in term_variants
                and len(term_variants[term_score_index[i][0]])>0
            )
            else 0
            for i,x in enumerate(importance_scores)
        ]

        # Renormalize
        if sum(importance_scores) == 0:
            return []  # avoid division by 0 error

        importance_scores = [
            x/sum(importance_scores)
            for x in importance_scores
        ]

        # Resize num_replacements to avoid p-sampling errors
        nonzero_entries = sum([x>0. for x in importance_scores])
        if num_replacements > nonzero_entries:
            num_replacements = nonzero_entries

        '''
        # DEBUG
        # Create a List of Lists of all variants
        candidate_variants = [
            v+[k]
            for k,v in term_variants.items()
        ]

        # Check the total number of variants
        candidate_sents = list(
            itertools.product(*candidate_variants)
        )

        # Set number of output variants to the total possible
        if len(candidate_sents) < num_output_sents:
            num_output_sents = len(candidate_sents)
        '''

        max_attempts = 50
        counter = 0
        new_sentences = set()
        while len(new_sentences) < num_output_sents:
            if counter > max_attempts:
                break

            # Select terms to replace
            rnd_indices = np.random.choice(
                len(term_score_index),
                size=num_replacements,
                replace=False,
                p=importance_scores
            )
            replace_terms = [
                term_score_index[i][0]
                for i in rnd_indices
            ]
        
            # Create List of Lists of term variants
            term_combinations = [
                term_variants.get(x[0], [x[0]])
                if x[0] in replace_terms
                else [x[0]]
                for i,x in enumerate(term_score_index)
            ]
        
            # Generate combinatorial variants
            candidate_sents = list(
                itertools.product(*term_combinations)
            )

            for sent in candidate_sents:
                new_sentences.add(' '.join(sent))
            counter += 1
            
        # Shuffle permutations, sanitize and slice
        new_sentences = list(new_sentences)
        random.shuffle(new_sentences)
        new_sentences = [
            re.sub(r'([A-Za-z0-9])(\s+)([^A-Za-z0-9])', r'\1\3',
                    x.replace('\' s ','\'s ')
            )
            for x in new_sentences[:num_output_sents]
        ]

        if len(new_sentences) < num_output_sents:
            logger.debug(
                '{}:replace_terms: unable to generate num_output_sents - {} of ({})'.format(
                    __file__.split('/')[-1],
                    len(new_sentences),
                    num_output_sents
                )
            )
        return new_sentences


if __name__ == '__main__':
    sent = 'what developmental network was discontinued after the shutdown of abc1 ?'
    sc = [('what', 13.653033256530762), ('developmental', 258.72607421875), ('network', 157.8809356689453), ('was', 18.151954650878906), ('discontinued', 30.241737365722656), ('after', 70.61669921875), ('the', 4.491329193115234), ('shutdown', 32.54951477050781), ('of', 11.050531387329102), ('abc1', 54.5350456237793), ('?', 0)]
    print('===> Orig: \n', sent, '\n')
    dw = DropTerms()
    print(dw.drop_terms(sent, 2, 4))
    '''
    misspellings = ReplaceTerms(rep_type='misspelling')
    synonyms = ReplaceTerms(rep_type='synonym')
    print('mispellings: ')
    print(misspellings.replace_terms(sent,importance_scores=sc, num_replacements=1,num_output_sents=5, sampling_strategy='topK',sampling_k=5))
    print('syns: ')
    print(synonyms.replace_terms(sent, importance_scores=sc, num_output_sents=10, sampling_k=5))
    '''
