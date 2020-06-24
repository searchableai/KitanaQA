import pkg_resources
import re
import nltk
import numpy as np
import math
from typing import List
from numpy import dot
from numpy.linalg import norm
from stop_words import get_stop_words
import nltk
from nltk.corpus import stopwords, wordnet
#from nlp_utils.firstnames import firstnames
#from doggmentator import get_logger
import logging
import pickle
import random

# init logging
logger = logging.getLogger('debug')

# stopwords and common names lists
stop_words = list(get_stop_words('en'))  # have around 900 stopwords
nltk_words = list(stopwords.words('english'))  # have around 150 stopwords
stop_words.extend(nltk_words)
stop_words = list(set(stop_words))
remove_list = stop_words
remove_list = [x.lower() for x in remove_list]

# load constrained word vectors
# Counter-fitting Word Vectors to Linguistic Constraints
# https://arxiv.org/pdf/1603.00892.pdf
#data_file = pkg_resources.resource_filename(
#    'doggmentator', 'support/glove.txt')

data_file = '/home/abijith/Desktop/searchable.ai/Doggmentor/Doggmentator/src/doggmentator/support/glove.txt'

logger.debug(
    '{}: loading pkg data {}'.format(
        __file__.split('/')[-1], data_file)
    )

vecs = {}
f = open(data_file, 'r')
for n, line in enumerate(f):
    line = line.strip().split()
    vecs[line[0].lower()] = np.asarray([float(x) for x in line[1:]])


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


def _wordnet_syns(term: str, num_syns: int=1) -> List:
    """Find synonyms using WordNet"""
    synonyms = []
    for syn in wordnet.synsets(term):
        for lemma in syn.lemmas() :
            lemma_name = ' '.join([x.lower() for x in lemma.name().split('_')])
            if lemma_name not in synonyms and lemma_name != term:
                synonyms.append(lemma_name)
    rand_idx = np.random.choice(np.arange(len(synonyms)), size=num_syns)
    return [synonyms[x] for x in rand_idx][0]


def _wordvec_syns(term: str, num_syns: int=10) -> List:
    """Find synonyms using word vectors"""

    # get word vector
    search_vector = vecs[term]

    # filter vectors
    vspace = [w for w in vecs.items() if w[0] != term]

    # sort (desc) vectors by similarity score
    word_dict = {
        x[0]: _cosine_similarity(x[1], search_vector) for x in vspace}
    vspace = [(x[0], word_dict[x[0]]) for x in vspace]
    vspace = sorted(vspace, key=lambda w: w[1], reverse=True)

    # filter vspace by threshold
    sim_thre = 0.55
    vspace = [x for x in vspace if x[1] >= sim_thre]
    if not vspace:
        return ''

    # choose random synonym
    rand_idx = np.random.choice(np.arange(len(vspace)), size=num_syns)
    synonyms = [vspace[x][0] for x in rand_idx]
    #print(synonyms)
    return synonyms


def gen_synonyms(
        sentence: str,
        rep_rate: float = 0.1,
        mode: str = 'wordvec') -> dict:
    """Generate terms similar to an input string using cosine similarity"""

    # sanity check and sanitize
    sent = _check_sent(sentence)
    terms = [x for x in sent.split() if x not in remove_list]

    if not terms:
        logger.error(
            '{}:_gen_synonyms: input {} is null or invalid'.format(
                __file__.split('/')[-1], terms
            )
        )
        return None
    elif mode not in ['wordnet', 'wordvec']:
        logger.error(
            '{}:_gen_synonyms: mode {} not supported'.format(
                __file__.split('/')[-1], terms
            )
        )
        return None

    # choose random term subset
    n_terms = math.ceil(rep_rate * len(terms))
    print('n_terms = ', n_terms)

    term_map = {x: x for x in terms}
    print('Term Map - ', term_map)
    syn_map = {}
    for term in terms:
        if term and term in vecs:
            print('term - ', term)
            if mode == 'wordvec':
                synonyms = _wordvec_syns(term)
            elif mode == 'wordnet':
                synonyms = _wordnet_syns(term)

            logger.debug(
                '{}:_gen_synonyms: {} - {}'.format(
                    __file__.split('/')[-1], term, synonyms
                )
            )

            if synonyms:
                syn_map[term] = synonyms
        else:
            logger.error(
                '{}:_gen_synonyms: Input {} not found in loaded vectors'.format(
                    __file__.split('/')[-1], term
                )
            )

    # check if any synonyms were found
    if not syn_map:
        return None
    else:
        # make sure n_terms doesn't exceed the available num syns
        if len(syn_map) < n_terms:
            n_terms = len(syn_map)

        # choose n_terms synonyms to replace
        #rand_idx = np.random.choice(
        #    np.arange(len(syn_map)), size=n_terms, replace=False)

        #syn_map = {
        #    x[1]: x[0] for n, x in enumerate(syn_map.items())
        #    if n in rand_idx
        #}

        return syn_map # Modified to return a synonym map for the input string

            
def replace_synonyms(input_str, word_score_tuple_list, K, sampling_strategy = 'topK', num_unique_samples = 3) -> str:

    # POS Tagging to Mask Tokens
    original_input_tokens = nltk.word_tokenize(input_str)
    pos_tagged_list_of_tuples = nltk.pos_tag(original_input_tokens)

    # [(name, 'NNP'), ..... ]
    masked_list = []
    for tup in pos_tagged_list_of_tuples:
        word, pos = tup
        if(pos == 'NNP'):
            if word.lower() not in masked_list:
                masked_list.append(word.lower())

    # Input tokens
    print()
    input_tokens = nltk.word_tokenize(input_str)
    input_tokens  = [x.lower() for x in input_tokens]

    # Sort by importance score
    if(sampling_strategy == 'topK'):
        word_score_tuple_list.sort(key=lambda x: x[1], reverse = True)
    elif(sampling_strategy == 'bottomK'):
        word_score_tuple_list.sort(key=lambda x: x[1], reverse = False)
    elif(sampling_strategy == 'randomK'):
        random.shuffle(word_score_tuple_list)

    #Get Synonym Map for input string
    syn_map = gen_synonyms(input_str)

    list_unique_samples = []
    for i in range(num_unique_samples):
        k = K
        copy_input_token = input_tokens.copy()
        for tup in (word_score_tuple_list):
            word, score = tup
            if(k > 0):
                if(word in masked_list):
                    pass
                else:
                    if word in copy_input_token:
                        if(word in syn_map.keys()):
                            for i in range(len(copy_input_token)):
                                if(copy_input_token[i] == word):
                                    index = random.choice(range(len(syn_map[word])))
                                    syn_word = syn_map[word][index]
                                    copy_input_token[i] = syn_word
                                    k -= 1

        for i in range(len(original_input_tokens)):
            if(copy_input_token[i] == original_input_tokens[i].lower()):
                copy_input_token[i] = original_input_tokens[i]
        modified_string = ' '.join([token for token in copy_input_token])

        if modified_string not in list_unique_samples:
            list_unique_samples.append(modified_string)

    return list_unique_samples


if __name__ == '__main__':
    #pass
    with open('/home/abijith/Downloads/SQuAD_v1.1_dev.pickle', mode='rb') as file:
        important_score_dict = pickle.load(file)

    K = 4
    id = 0
    input_str  = 'How many Asus gp3 molecules leave in the fission cycle ?'
    word_score_tuple_list = important_score_dict[id]

    syn_rep_str = replace_synonyms(input_str, word_score_tuple_list, K, sampling_strategy='topK', num_unique_samples = 3)
    print()
    print('I/P - ', input_str)
    print('O/P - ', syn_rep_str)
    print()
    