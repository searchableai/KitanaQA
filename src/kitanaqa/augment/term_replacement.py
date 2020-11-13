import pkg_resources
import random
import itertools
import re
import json
import numpy as np
from stop_words import get_stop_words
from typing import List, Dict, Tuple
from numpy import dot
from numpy.linalg import norm

import nltk
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords

try:
    import sparknlp
    from sparknlp.pretrained import PretrainedPipeline
    from sparknlp.annotator import *
    from sparknlp.common import RegexRule
    from sparknlp.base import *
    SPARK_NLP_ENABLED = True
except Exception as e:
    SPARK_NLP_ENABLED = False

from kitanaqa.augment.generators import SynonymReplace, MisspReplace, MLMSynonymReplace
from kitanaqa import get_logger

# init logging
logger = get_logger()

# stopwords and common names lists
stop_words = list(get_stop_words('en'))  # have around 900 stopwords
nltk_words = list(stopwords.words('english'))  # have around 150 stopwords
stop_words.extend(nltk_words)
stop_words = list(set(stop_words))
remove_list = [x.lower() for x in stop_words]


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
        scores: List[Tuple]=None,
        remove_stop: bool=True) -> List[Tuple]:
    """ Initialize and sanitize importance scores """

    # Check mode parameters
    if scores and mode not in ['topK', 'bottomK']:
        mode = 'topK'
    if mode in ['topK', 'bottomK']:
        if not mode_k:
            mode_k = 10

    if not scores:
        # Uniform initialization
        if remove_stop:
            scores = [
                1
                if x not in remove_list
                else 0
                for x in tokens
            ]
        else:
            scores = [1 for x in tokens]
        
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


class RepeatTerms():
    """ A class to generate sentence perturbations by repeating target terms
    ...
    Methods
    ----------
    repeat_terms(sentence, num_terms, num_output_sents)
      Generate synonyms for an input term 
    """
    def __init__(self, use_stop: bool=True):
        """Instantiate a ReplaceTerms object
        Parameters
        ----------
        use_stop : Optional(bool)
            Flag specifying whether to only apply perturbation to stopwords. If True, when calculating the sampling weights for any perturbation, non-stopwords will be zeroed. The default value is True.
        """
        self.use_stop = use_stop

    def repeat_terms(
            self,
            sentence: str,
            num_terms: int=1,
            num_output_sents: int=1) -> List:
        """Generate a certain number of sentence perturbations by repeating select terms

        Parameters
        ----------
        sentence : str
            The input sentence to be perturbed.
        num_terms : Optional(int)
            The number of terms to target in the original sentence, with this perturbation. The number should be greater than 0. The default value is 1.
        num_output_sents : Optional(int)
            The number of perturbed sentences to produce. The number should be greater than 0. The default is 1.

        Returns
        -------
        [str]
            Returns a list of perturbed sentences for the input sentence.

        Example
        -------
        >>> from term_replacement import RepeatTerms
        >>> p = RepeatTerms()
        >>> sent = "I was born in a small town"
        >>> num_terms = 1
        >>> num_output_sents = 1
        >>> p.generate(sent, num_terms, num_output_sents)
        ['I was was born in a small town']
        """

        inputs = validate_inputs(
            num_terms,
            num_output_sents)
        num_terms = inputs.pop(0)
        num_output_sents = inputs.pop(0)

        # Whitespace tokenizer
        word_tokens = word_tokenize(sentence)

        # Create list of candidate repeat words
        repeat_word_indices = []
        for idx, word in enumerate(word_tokens):
            if self.use_stop:
                if word in remove_list:
                    repeat_word_indices.append(idx)
            else:
                repeat_word_indices.append(idx)

        # Ensure num_terms does not exceed possible terms
        if num_terms > len(repeat_word_indices):
            num_terms = len(repeat_word_indices)

        new_sentences = []
        if len(repeat_word_indices) == 0:
            return new_sentences

        # Randomly choose num_terms indices from all repeat_word_indices
        comb = []
        if num_terms == -1:
            # Generate all possible combinations for debugging
            for r in range(1, len(repeat_word_indices) + 1):
                comb += list(itertools.combinations(repeat_word_indices, r))
        else:
            # Generate all combinations of num_terms stopwords
            comb = list(itertools.combinations(repeat_word_indices, num_terms))

        # Randomly sample repeat term combinations
        n_chosen_indices = [comb[idx] for idx in np.random.choice(len(comb), size=min(num_output_sents, len(comb)), replace=False)]
        for chosen_indices in n_chosen_indices:
            new_words = [
                    [word_tokens[idx]]
                    if idx not in chosen_indices
                    else [word_tokens[idx], word_tokens[idx]]
                    for idx in range(len(word_tokens))
            ]
            new_words = [sub for subl in new_words for sub in subl]
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
                '{}:repeat_terms: unable to generate num_output_sents - {} of ({})'.format(
                    __file__.split('/')[-1],
                    len(new_sentences),
                    num_output_sents
                )
            )
        return new_sentences


class DropTerms():
    """ A class to generate sentence perturbations by dropping target terms
    ...
    Methods
    ----------
    drop_terms(sentence, num_terms, num_output_sents)
      Generate synonyms for an input term 
    """
    def __init__(self, use_stop: bool=True):
        """Instantiate a DropTerms object
        Parameters
        ----------
        use_stop : Optional(bool)
            Flag specifying whether to only apply perturbation to stopwords. If True, when calculating the sampling weights for any perturbation, non-stopwords will be zeroed. The default value is True.
        """
        self.use_stop = use_stop

    def drop_terms(
            self,
            sentence: str,
            num_terms: int=1,
            num_output_sents: int=1) -> List:
        """Generate a certain number of sentence perturbations by dropping select terms

        Parameters
        ----------
        sentence : str
            The input sentence to be perturbed.
        num_terms : Optional(int)
            The number of terms to target in the original sentence, with this perturbation. The number should be greater than 0. The default value is 1.
        num_output_sents : Optional(int)
            The number of perturbed sentences to produce. The number should be greater than 0. The default is 1.

        Returns
        -------
        [str]
            Returns a list of perturbed sentences for the input sentence.

        Example
        -------
        >>> from term_replacement import DropTerms
        >>> p = DropTerms()
        >>> sent = "I was born in a small town"
        >>> num_terms = 3
        >>> num_output_sents = 1
        >>> p.generate(sent, num_terms, num_output_sents)
        ['I born small town']
        """

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
            if self.use_stop:
                if word in remove_list:
                    drop_word_indices.append(idx)
            else:
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
    """ A class to generate sentence perturbations by replacement
    ...
    Methods
    ----------
    replace_terms(sentence, importance_scores, num_replacements, num_output_sents, sampling_strategy, sampling_k)
      Generate synonyms for an input term 
    """
    global SPARK_NLP_ENABLED

    def __init__(
            self,
            rep_type: str='synonym',
            use_ner: bool=True):
        """Instantiate a ReplaceTerms object

        Parameters
        ----------
        rep_type : Optional(str)
            The type of target perturbation. May include `synonym` for word2vec replacement, `mlmsynonym` for MLM-based replacement, or `misspelling` for misspelling replacement.
        use_ner : Optional(bool)
            Flag specifying whether to use entity-aware replacement. If True, when calculating the sampling weights for any perturbation, named entities will be zeroed. In this case, the NER model is loaded here. The default value is True.
        """
        self.use_ner = use_ner if SPARK_NLP_ENABLED else False
        self.rep_type = rep_type
        if rep_type not in ['synonym', 'misspelling', 'mlmsynonym']:
            logger.error(
                '{}:ReplaceTerms __init__ invalid rep_type'.format(
                    __file__.split('/')[-1]
                )
            )
            raise ValueError('Not an accepted generator type')
        self._generator = self._get_generator(rep_type)
        if not self._generator:
            raise RuntimeError('Unable to init generator')
        if self.use_ner:
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

        if self.use_ner:
            # Use spark-nlp tokenizer for entity-aware mask
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
        else:
            # Use simple NLTK tokenizer
            toks = nltk.word_tokenize(sentence)
            mask = [0]*len(toks)
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
        elif name == 'mlmsynonym':
            try:
                _syn = MLMSynonymReplace()
                return _syn
            except Exception as e:
                logger.error(
                    '{}:replace_terms: unable to load word vectors'.format(
                        __file__.split('/')[-1]
                    )
                )
        return

    def replace_terms(
            self,
            sentence: str,
            importance_scores: List=None,
            num_replacements: int=1,
            num_output_sents: int=1,
            sampling_strategy: str='random',
            sampling_k: int=None) -> List:
        """Generate a certain number of sentence perturbations by replacement using either misspelling or synonyms

        Parameters
        ----------
        sentence : str
            The input sentence to be perturbed.
        importance_scores : Optional(List)
            List of tuples defining a weight for each term in the tokenized sentence. These weights are used during sampling to influence perturnation probabilities. If None, uniform sampling is used by default.
        num_replacements : Optional(int)
            Target number of terms to replace in the original sentence. The number is chosen randomly using the target as an upper bound, and lower bound of 1. The default is 1.
        num_output_sents : Optional(int)
            Target number of perturbed sentences to generate based on the original sentence. The default is 1.
        sampling_strategy : Optional(str)
            Strategy used to sample terms to perturb in the original sentence. The default is random. If importance_scores is given, then sampling_strategy may be `topK` or `bottomK`, in which case the importance_scores (or inverted scores) vector is used for weighted sampling.
        sampling_k : Optional(int)
            The number of terms in the importance score vector to include in topK or bottomK sampling. This parameter is not used by the default sampling_strategy, `random` sampling.
        Returns
        -------
        [str]
            Returns a list of perturbed sentences for the input sentence.

        Example
        -------
        >>> from term_replacement import ReplaceTerms
        >>> p = ReplaceTerms(rep_type="synonym")
        >>> sent = "I was born in a small town"
        >>> num_terms = 1
        >>> num_output_sents = 1
        >>> p.generate(sent, num_terms, num_output_sents)
        ['I born in a small village']

        >>> from term_replacement import ReplaceTerms
        >>> p = ReplaceTerms(rep_type="misspelling")
        >>> sent = "I was born in a small town"
        >>> num_terms = 1
        >>> num_output_sents = 1
        >>> p.generate(sent, num_terms, num_output_sents)
        ['I born in a smal town']
        """

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

        if self.rep_type == 'misspelling':
            remove_stop = True
        else:
            remove_stop = False

        # Initialize sampling scores
        importance_scores = get_scores(
            tokens,
            sampling_strategy,
            sampling_k,
            importance_scores,
            remove_stop)

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
        generated = {x[0]:None for x in term_score_index}
        generated = {
            x[0]:self._generator.generate(x[0].lower(), 10, **{'toks':tokens, 'token_idx':i})
            for i,x in enumerate(term_score_index)
            if not x[2]
        }

        term_variants = {
            x[0]:generated.get(x[0], [])
            if (i in rep_term_indices
                and not masked_vector[i])
            else []
            for i,x in enumerate(term_score_index)
        }

        # Check if there are enough candidate terms
        if not term_variants:
            logger.warning(
                '{}:replace_terms: unable to generate num_variants - {} of ({})'.format(
                    __file__.split('/')[-1],
                        num_replacements,
                        len(term_variants) - sum(masked_vector)
                    )
                )
        else:
            term_variants = {
                k:[x[0].upper()+x[1:] for x in v]
                if k[0].isupper()
                else v
                for k,v in term_variants.items()
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
        if not term_variants or len([x[2]==0 for x in term_score_index])==0:
            raise Exception('no term variants or term_score_index')

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
        new_sentences = [x for x in new_sentences if x != sentence]

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
