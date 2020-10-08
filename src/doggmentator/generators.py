import pkg_resources
import sparknlp
import re
import random
import itertools
import re
import json
import numpy as np
from typing import List, Dict, Tuple
from numpy import dot
from numpy.linalg import norm
from stop_words import get_stop_words
import nltk
from nltk.corpus import stopwords, wordnet
from sparknlp.pretrained import PretrainedPipeline
from sparknlp.annotator import *
from sparknlp.common import RegexRule
from sparknlp.base import *
from doggmentator.nlp_utils.firstnames import firstnames
from doggmentator import get_logger

# init logging
logger = get_logger()

# stopwords and common names lists
stop_words = list(get_stop_words('en'))  # have around 900 stopwords
nltk_words = list(stopwords.words('english'))  # have around 150 stopwords
stop_words.extend(nltk_words)
stop_words = list(set(stop_words))
remove_list = firstnames+stop_words
remove_list = [x.lower() for x in remove_list]


class BaseGenerator:
    """ Base class for generator objects """
    def _check_sent(self, sent: str) -> str:
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
            self,
            v1: np.ndarray,
            v2: np.ndarray) -> float:
        """ Calculate cosine similarity between two vectors """
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


class MisspReplace(BaseGenerator):
    
    """ A class to replace commonly misspelled terms
    ...
    Methods
    ----------
    generate(term, num_missp)
      Generate misspellings for an input term 
    """
    def __init__(self):
        super().__init__()
        self._missp = None
        self._load_misspellings()

    def _load_misspellings(self):
        """ 
        Load dict of term misspellings
        ref:    wiki, brikbeck
        """
        data_file = pkg_resources.resource_filename(
            'doggmentator', 'support/missp.json')
        logger.debug(
            '{}: loading pkg data {}'.format(
                __file__.split('/')[-1], data_file)
            )
        with open(data_file, 'r') as f:
            self._missp = json.load(f)

    def generate(
            self,
            term: str,
            num_missp: int=10) -> List:
        
        """Generate a certain number of misspellings for the input term. The misspelling 

        Parameters
        ----------
        term : str
            The input term for which we are looking for misspellings.
        num_missp : Optional(int)
            The number of misspellings for the input term. The number of misspelling should be greater than 1. The default value is 10.

        Returns
        -------
        [str]
            Returns a list of misspelling if there is any misspelling for the input term. Otherwise an empty list
            
        Example
        -------
        >>> from generators import MisspReplace
        >>> mr = MisspReplace()
        >>> term = "worried"
        >>> num_missp = 5
        >>> mr.generate(term, num_missp)
        ['woried', 'worred']
        """

        # Num misspellings must be gte 1
        num_missp = max(num_missp, 1)
        import pdb; pdb.set_trace()
        if term in self._missp:
            return self._missp[term][:num_missp]
        else:
            return []


class SynonymReplace(BaseGenerator):
    """ Find synonyms using word vectors """
    def __init__(self):
        super().__init__()
        self._vecs = None
        self._load_embeddings()

    def _load_embeddings(self):
        """ 
        Load constrained word vectors
        ref:    Counter-fitting Word Vectors to Linguistic Constraints
                https://arxiv.org/pdf/1603.00892.pdf
        """
        data_file = pkg_resources.resource_filename(
            'doggmentator', 'support/counter-fitted-vectors.txt')
            #'doggmentator', 'support/paragram.txt')
            #'doggmentator', 'support/glove.txt')

        logger.debug(
            '{}: loading pkg data {}'.format(
                __file__.split('/')[-1], data_file)
            )
        vecs = {}
        f = open(data_file, 'r')
        for n, line in enumerate(f):
            line = line.strip().split()
            vecs[line[0].lower().strip()] = np.asarray([float(x) for x in line[1:]])
        self._vecs = vecs

    def generate(
            self,
            term: str,
            num_syns: int=10,
            similarity_thre: float=0.7) -> List:
        """ Generate synonyms for an input term """

        # Number of synonyms must be gte 1
        num_syns = max(num_syns, 1)

        if term in self._vecs:
            search_vector = self._vecs[term]
        else:
            return []

        # Filter vectors
        vspace = [w for w in self._vecs.items() if w[0] != term]

        # Sort (desc) vectors by similarity score
        word_dict = {
            x[0]: self._cosine_similarity(x[1], search_vector) for x in vspace}
        vspace = sorted(list(word_dict.items()), key=lambda w: w[1], reverse=True)

        # Filter vspace by threshold
        vspace = [x for x in vspace if x[1] >= similarity_thre]
        if not vspace:
            return []

        # Choose top synonyms
        synonyms = [x[0] for x in vspace[:num_syns]]
        return synonyms
