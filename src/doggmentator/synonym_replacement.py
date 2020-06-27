import pkg_resources
import logging
import time
import sys
import sparknlp
import re
import random
import nltk
import pickle
import math
import itertools
import re
import numpy as np
from typing import List, Dict, Tuple
from numpy import dot
from numpy.linalg import norm
from stop_words import get_stop_words
from nltk.corpus import stopwords, wordnet
from nlp_utils.firstnames import firstnames
from doggmentator import get_logger
from sparknlp.pretrained import PretrainedPipeline
from sparknlp.annotator import *
from sparknlp.common import RegexRule
from sparknlp.base import *

# init logging
logger = get_logger()

# stopwords and common names lists
stop_words = list(get_stop_words('en'))  # have around 900 stopwords
nltk_words = list(stopwords.words('english'))  # have around 150 stopwords
stop_words.extend(nltk_words)
stop_words = list(set(stop_words))
remove_list = firstnames+stop_words
remove_list = [x.lower() for x in remove_list]


def _check_sent(sent: str) -> str:
    """Run sanity checks on input and sanitize"""
    try:
        sent = str(sent)
    except Exception as e:
        logger.error(
            '{}:_check_sent: {} - {}'.format(
                __file__.split('/')[-1], sent, e)
            )
        return ''
    sent = re.sub(r'[^A-Za-z0-9.\' ]', '', sent).lower()
    return sent


def _cosine_similarity(
        v1: np.ndarray,
        v2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return dot(v1, v2) / (norm(v1) * norm(v2))


def _wordnet_syns(term: str, num_syns: int=10) -> List:
    """Find synonyms using WordNet"""
    synonyms = []
    for syn in wordnet.synsets(term):
        for lemma in syn.lemmas() :
            lemma_name = ' '.join([x.lower() for x in lemma.name().split('_')])
            if lemma_name not in synonyms and lemma_name != term:
                synonyms.append(lemma_name)
    rand_idx = np.random.choice(len(synonyms), size=num_syns)
    return [synonyms[x] for x in rand_idx][0]


class WordVecSyns:
    """Find synonyms using word vectors"""
    def __init__(self):
        self._vecs = self._load_embeddings()

    def _load_embeddings(self):
        """ 
        Load constrained word vectors
        ref:    Counter-fitting Word Vectors to Linguistic Constraints
                https://arxiv.org/pdf/1603.00892.pdf
        """
        data_file = pkg_resources.resource_filename(
            'doggmentator', 'support/glove.txt')
        logger.debug(
            '{}: loading pkg data {}'.format(
                __file__.split('/')[-1], data_file)
            )
        vecs = {}
        f = open(data_file, 'r')
        for n, line in enumerate(f):
            line = line.strip().split()
            vecs[line[0].lower().strip()] = np.asarray([float(x) for x in line[1:]])
        self.vecs = vecs

    def get_synonyms(
            self,
            term: str,
            num_syns: int=10,
            similarity_thre: float=0.7) -> List:
        """ Generate synonyms for an input term """
        if term in self.vecs:
            search_vector = self.vecs[term]
        else:
            return []

        # Filter vectors
        vspace = [w for w in self.vecs.items() if w[0] != term]

        # Sort (desc) vectors by similarity score
        word_dict = {
            x[0]: _cosine_similarity(x[1], search_vector) for x in vspace}
        vspace = [(x[0], word_dict[x[0]]) for x in vspace]
        vspace = sorted(vspace, key=lambda w: w[1], reverse=True)

        # Filter vspace by threshold
        vspace = [x for x in vspace if x[1] >= similarity_thre]
        if not vspace:
            return

        # Choose top synonyms
        synonyms = [x[0] for x in vspace[:num_syns]]
        return synonyms


def get_entities(sentence: str) -> Dict:
    """ Tokenize and annotate sentence """

    # Start Spark session with Spark NLP
    allowed_tags = ['PER','LOC','ORG','MISC']
    spark = sparknlp.start()
    pipeline = PretrainedPipeline('recognize_entities_dl', lang='en')

    # Annotate your testing dataset
    result = pipeline.annotate(sentence)
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


def get_scores(
        tokens: List[str],
        scores: List[Tuple]=None) -> List[Tuple]:
    """ Initialize and sanitize importance scores """
    if not scores:
        # Random initialization
        scores = np.random.rand(
            len(tokens)
        )
        scores = [
            x
            if tokens[i] not in remove_list
            else 0
            for i,x in enumerate(scores)
        ]
        scores = [
            x/sum(scores)
            for x in scores
        ]
        scores = list(zip(tokens, scores))
    else:                         
        if len(scores) != len(tokens):
            logger.error(    
                '{}:get_scores: ({},{}) mismatch in entity vec and importance score dims'.format(
                    __file__.split('/')[-1],
                    len(tokens),
                    len(scores)
                )      
            ) 
            scores = None
        else:
            # Ensure score types, norm, sgn
            scores = [   
                (x[0],abs(float(x[1])))
                for x in scores
            ]
            scores = [
                (x[0],x[1]/sum(scores))
                for x in scores
            ]
    return scores


def replace_terms(
        sentence: str,
        rep_type: str='synonym',
        importance_scores: List=None,
        num_replacements: int=1,
        num_output_sents: int=1,
        sampling_strategy: str='random') -> List:
    """Generate list of strings by replacing specified number of terms with their synonyms"""

    # Extract entities in the input sentence to mask
    masked_vector, tokens = get_entities(sentence)

    # Initialize sampling scores
    importance_scores = get_scores(tokens, importance_scores)

    if not importance_scores:
        return

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

    # Invert scores if sampling least important terms
    if sampling_strategy == 'bottomK':
        importance_scores = [
            1/x if x>0 else 0
            for x in importance_scores
        ]

    # Renormalize
    importance_scores = [
        x/sum(importance_scores)
        for x in importance_scores
    ]

    # Candidate terms for synonym replacement
    rep_term_indices = [
        w[1] for w in term_score_index if not w[2]
    ]

    # Create List of Lists of term variants
    # TODO: test misspellings with support data
    if rep_type == 'misspelling':
        try:
            misspellings = _load_misspellings()
        except Exception as e:
            logger.error(
                '{}:replace_terms: unable to load misspellings'.format(
                    __file__.split('/')[-1]
                )
            )
            return

        term_variants = {
            x[0]:misspellings.get(x[0],x[0])
            if (i in rep_term_indices
                and not masked_vector[i])
            else x[0]
            for i,x in enumerate(term_score_index)
        }
    elif rep_type == 'synonym':
        wordvec = WordVecSyns()
        synonyms = {
            x[0]:wordvec.get_synonyms(x[0])
            for i,x in enumerate(term_score_index)
        }
        term_variants = {
            x[0]:synonyms.get(x[0])
            if (i in rep_term_indices
                and not masked_vector[i]
                and synonyms.get(x[0]))
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
    importance_scores = [
        x/sum(importance_scores)
        for x in importance_scores
    ]

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

    max_attempts = 10
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
        re.sub(r'([A-Za-z0-9])(\s+)([^A-Za-z0-9])', r'\1\3', x)
        for x in new_sentences[:num_output_sents]
    ]
    if len(new_sentences) < num_output_sents:
        logger.debug(
            '{}:replace_terms: unable to generate num_output_sents {}'.format(
                __file__.split('/')[-1],
                len(new_sentences)
            )
        )
    return new_sentences


if __name__ == '__main__':
    sent  = 'how many g3p molecules leave the cycle?'
    print(replace_terms(sent, num_output_sents=10, num_replacements=2))
