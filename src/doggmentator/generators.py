import pkg_resources
import sparknlp
import re
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
from stop_words import get_stop_words
from nltk.corpus import stopwords, wordnet
from sparknlp.pretrained import PretrainedPipeline
from sparknlp.annotator import *
from sparknlp.common import RegexRule
from sparknlp.base import *
from transformers import AutoTokenizer, BertForMaskedLM
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

    synonyms = []
    for syn in wordnet.synsets(term):
        for lemma in syn.lemmas() :
            lemma_name = ' '.join([x.lower() for x in lemma.name().split('_')])
            if lemma_name not in synonyms and lemma_name != term:
                synonyms.append(lemma_name)
    rand_idx = np.random.choice(len(synonyms), size=num_syns)
    return [synonyms[x] for x in rand_idx][0]


class MisspReplace(BaseGenerator):
    """ Replace commonly misspelled terms """
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
            toks: List=None,
            token_idx: int=None,
            num_missp: int=10) -> List:
        """ Generate misspellings for an input term """

        # Num misspellings must be gte 1
        num_missp = max(num_missp, 1)

        if term in self._missp:
            return self._missp[term][:num_missp]
        else:
            return []


class MLMSynonymReplace(BaseGenerator):
    """ Find synonym using MLM """
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
            toks: str,
            token_idx: int,
            num_syns: int=10,
        ):
        # Need to account for possible duplicate term in results
        num_syns += 1
        toks[token_idx] = self.tokenizer.mask_token
        sentence = ' '.join(toks)
        encoded_sent = self.tokenizer.encode(sentence, return_tensors="pt")
        mask_token_idx = torch.where(encoded_sent == self.tokenizer.mask_token_id)[1]
        token_logits = self.model(encoded_sent)[0]
        mask_token_logits = token_logits[0, mask_token_idx, :]

        probs = torch.nn.functional.softmax(mask_token_logits, dim=1)
        topk = torch.topk(probs, num_syns)
        top_n_probs, top_n_tokens = topk.values.detach().numpy()[0], topk.indices.detach().numpy()[0]
        results = [self.tokenizer.decode([top_n_tokens[n]]) for n in range(min(num_syns,len(top_n_probs)))]
        results = [x for x in results if x != term]
        return results


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
            toks: List=None,
            token_idx: int=None,
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
