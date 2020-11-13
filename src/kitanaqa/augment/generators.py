import pkg_resources
import re
import os
import random
import itertools
import re
import json
import torch
import nltk
import numpy as np
from typing import List, Dict, Tuple
from numpy import dot
from numpy.linalg import norm
from transformers import AutoTokenizer, BertForMaskedLM
from kitanaqa import get_logger

# init logging
logger = get_logger()


class BaseGenerator:
    """ A base class for generating token-level perturbations
    ...
    Methods
    ----------
    _check_sent(sent)
      Validate and sanitize an input sentence
    _cosine_similarity(v1, v2)
      Calculate the cosine similarity between two vectors
    """

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
        sent = ' '.join(sent.split())
        return sent

    def _cosine_similarity(
            self,
            v1: np.ndarray,
            v2: np.ndarray) -> float:
        """ Calculate cosine similarity between two vectors """
        return dot(v1, v2) / (norm(v1) * norm(v2))


def _wordnet_syns(term: str, num_syns: int=10) -> List:
    """Find synonyms using WordNet"""
    from warnings import warn
    warn("WordNet synonym generation is deprecated. Please use one of the other methods available.")

    from nltk.corpus import wordnet
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
    generate(term, num_target)
      Generate misspellings for an input term 
    """
    def __init__(self):
        super().__init__()
        self._missp = None
        self._load_misspellings()

    def _load_misspellings(self):
        """ 
        Load dict of term misspellings
        Source References:
            wiki (https://en.wikipedia.org/wiki/Wikipedia:Lists_of_common_misspellings)
            brikbeck (https://www.dcs.bbk.ac.uk/~ROGER/corpora.html)
        """
        data_file = pkg_resources.resource_filename(
            'kitanaqa', 'support/missp.json')
        logger.debug(
            '{}: loading pkg data {}'.format(
                __file__.split('/')[-1], data_file)
            )
        with open(data_file, 'r') as f:
            self._missp = json.load(f)

    def generate(
            self,
            term: str,
            num_target: int,
            **kwargs) -> List:
        """Generate a certain number of misspellings for the input term.

        Parameters
        ----------
        term : str
            The input term for which we are looking for misspellings.
        num_target : int
            The target number of misspellings to generate for the input term. The number of misspelling should be greater than 0
        kwargs : Dict
            A set of generator-specific arguments.

        Returns
        -------
        [str]
            Returns a list of misspellings if any. Otherwise an empty list
            
        Example
        -------
        >>> from generators import MisspReplace
        >>> mr = MisspReplace()
        >>> term = "worried"
        >>> num_target = 5
        >>> mr.generate(term=term, num_target=num_target)
        ['woried', 'worred']
        """

        # Num misspellings must be gte 1
        num_target = max(num_target, 1)

        if term in self._missp:
            return self._missp[term][:num_target]
        else:
            return []


class MLMSynonymReplace(BaseGenerator):
    """ A class to replace synonyms using a masked language model (MLM)
    ...
    Methods
    ----------
    generate(term, num_target, toks, token_idx)
      Generate misspellings for an input term 
    """ 
    def __init__(self):
        super().__init__()
        self.model_path = 'bert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, use_fast=True)
        self.model = BertForMaskedLM.from_pretrained(
            self.model_path)

    def generate(
            self,
            term: str,
            num_target: int,
            **kwargs) -> List:
        """Generate a certain number of synonyms using an MLM.

        Parameters
        ----------
        term : str
            The input term for which we are looking for misspellings.
        num_target : int
            The target number of synonyms to generate for the input term. The number should be greater than 0
        kwargs: Dict
            A set of generator-specific arguments
            - toks : List
                The tokenized source string containing the target term.
            - token_idx : int
                The index of the target string in the tokenized source string
            
        Returns
        -------
        [str]
            Returns a list of synonyms if any. Otherwise an empty list
            
        Example
        -------
        >>> from generators import MLMSynonymReplace
        >>> mr = MLMSynonymReplace()
        >>> term = "small"
        >>> toks = ['I', 'was', 'born', 'in', 'a', 'small', 'town']
        >>> token_idx = 5
        >>> num_target = 2
        >>> mr.generate(term=term, num_target=num_target, {'toks':toks, 'token_idx':token_idx})
        ['little', 'mining']
        """
        toks = kwargs.get('toks', None)
        token_idx = kwargs.get('token_idx', None)
        if not toks or not token_idx:
            raise RuntimeError('Input parameters `toks` and `token_idx` must be specified when using MLM generator')

        # Need to account for possible duplicate term in results
        num_target += 1
        toks[token_idx] = self.tokenizer.mask_token
        sentence = ' '.join(toks)
        encoded_sent = self.tokenizer.encode(sentence, return_tensors="pt")
        mask_token_idx = torch.where(encoded_sent == self.tokenizer.mask_token_id)[1]
        token_logits = self.model(encoded_sent)[0]
        mask_token_logits = token_logits[0, mask_token_idx, :]

        probs = torch.nn.functional.softmax(mask_token_logits, dim=1)
        topk = torch.topk(probs, num_target)
        top_n_probs, top_n_tokens = topk.values.detach().numpy()[0], topk.indices.detach().numpy()[0]
        results = [self.tokenizer.decode([top_n_tokens[n]]) for n in range(min(num_target,len(top_n_probs)))]
        results = [x for x in results if x != term]
        return results


class SynonymReplace(BaseGenerator):
    """ A class to generate synonyms for an input term using word2vec
    ...
    Methods
    ----------
    generate(term, num_targets, {'similarity_thre':0.5})
      Generate synonyms for an input term 
    """
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
            'kitanaqa', 'support/counter-fitted-vectors.txt')
        if not os.path.isfile(data_file):
            logger.info('Extracting word vectors...')
            import zipfile
            with zipfile.ZipFile(data_file+'.zip',"r") as zip_f:
                outfile = '/'.join(data_file.split('/')[:-1])
                zip_f.extractall(outfile)

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
            num_target: int,
            **kwargs) -> List:
        """Generate a certain number of synonyms using a word2vec model

        Parameters
        ----------
        term : str
            The input term for which we are looking for misspellings.
        num_target : int
            The target number of synonyms to generate for the input term. The number should be greater than 0
        kwargs: Optional(Dict)
            A set of generator-specific arguments
            - similarity_thre : Optional(float)
                Threshold of cosine similarity values in generated terms. The default value is 0.7
            
        Returns
        -------
        [str]
            Returns a list of synonyms if any. Otherwise an empty list
            
        Example
        -------
        >>> from generators import SynonymReplace
        >>> mr = SynonymReplace()
        >>> term = "worried"
        >>> num_target = 3
        >>> similarity_thre = 0.7
        >>> sr.generate(term, num_target, {'similarity_thre': 0.7})
        ['apprehensive', 'preoccupied', 'worry']
        """

        similarity_thre = kwargs.get('similarity_thre', 0.7)

        # Number of synonyms must be gte 1
        num_target = max(num_target, 1)

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
        synonyms = [x[0] for x in vspace[:num_target]]
        return synonyms
