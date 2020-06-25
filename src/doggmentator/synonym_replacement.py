import pkg_resources
import re
import nltk
import numpy as np
import math
from typing import List
from numpy import dot
from numpy.linalg import norm
from stop_words import get_stop_words
from nltk.corpus import stopwords, wordnet
from nlp_utils.firstnames import firstnames
from doggmentator import get_logger
import logging
# For Spark NLP
import sys
import time
#Spark NLP
import sparknlp
from sparknlp.pretrained import PretrainedPipeline
from sparknlp.annotator import *
from sparknlp.common import RegexRule
from sparknlp.base import *
import random
import pickle



# init logging
logger = get_logger()


# stopwords and common names lists
stop_words = list(get_stop_words('en'))  # have around 900 stopwords
nltk_words = list(stopwords.words('english'))  # have around 150 stopwords
stop_words.extend(nltk_words)
stop_words = list(set(stop_words))
remove_list = firstnames+stop_words
remove_list = [x.lower() for x in remove_list]

# load constrained word vectors
# Counter-fitting Word Vectors to Linguistic Constraints
# https://arxiv.org/pdf/1603.00892.pdf
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

# Output is a string and 'num_syns' is not used just returns the first element of a singleton list
def _wordvec_syns(term: str, num_syns: int=1) -> List:
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
    sim_thre = 0.7
    vspace = [x for x in vspace if x[1] >= sim_thre]
    if not vspace:
        return ''

    # choose random synonym
    rand_idx = np.random.choice(np.arange(len(vspace)), size=1)
    synonym = [vspace[x][0] for x in rand_idx][0]
    return synonym

# Modified _wordvec_syns to return a list of synonyms equal to 'num_syns' parameter
## Same with 'WordNet' but have modifying that

def _wordvec_list_syns(term: str, num_syns: int=1) -> List:
    """Find synonyms using word vectors"""
    # get word vector
    try:
        search_vector = vecs[term]
    except:
        return ''

    # filter vectors
    vspace = [w for w in vecs.items() if w[0] != term]

    # sort (desc) vectors by similarity score
    word_dict = {
        x[0]: _cosine_similarity(x[1], search_vector) for x in vspace}
    vspace = [(x[0], word_dict[x[0]]) for x in vspace]
    vspace = sorted(vspace, key=lambda w: w[1], reverse=True)

    # filter vspace by threshold
    sim_thre = 0.7
    vspace = [x for x in vspace if x[1] >= sim_thre]
    if not vspace:
        return ''

    # choose random synonym
    rand_idx = np.random.choice(np.arange(len(vspace)), size=num_syns)
    synonyms = [vspace[x][0] for x in rand_idx]
    return synonyms


def gen_synonyms(
        sentence: str,
        rep_rate: float = 0.1,
        mode: str = 'wordvec') -> str:
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

    term_map = {x: x for x in terms}
    syn_map = {}
    for term in terms:
        if term and term in vecs:
            if mode == 'wordvec':
                synonym = _wordvec_syns(term)
            elif mode == 'wordnet':
                synonym = _wordnet_syns(term)

            logger.debug(
                '{}:_gen_synonyms: {} - {}'.format(
                    __file__.split('/')[-1], term, synonym
                )
            )

            if synonym:
                syn_map[synonym] = term
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
        rand_idx = np.random.choice(
            np.arange(len(syn_map)), size=n_terms, replace=False)

        syn_map = {
            x[1]: x[0] for n, x in enumerate(syn_map.items())
            if n in rand_idx
        }

        # replace terms in the original string
        syn_terms = ' '.join(
            [syn_map[x] if x in syn_map else x for x in sent.split()])

        if syn_terms == sent:
            return None
        else:
            return syn_terms

def get_entities(
        sentence: str) -> dict:
    spark = sparknlp.start()
    pipeline = PretrainedPipeline('recognize_entities_dl', lang='en')
    result = pipeline.annotate(sentence)
    return result



def replace_with_synonyms(
        sentence: str,
        word_score_tuple_list: list,
        num_terms_to_replace: int = 1,
        sampling_strategy: str = 'topK',
        num_output_samples: int = 1,
        mode: str = 'wordvec') -> List:
    """Generate list of strings by replacing specified number of terms with their synonyms"""

    # extract entities in the input sentence to mask
    result_dict = get_entities(sentence)
    tokens = result_dict['token']
    masked_vector = [1 if (ner).startswith('I') else 0 for ner in result_dict['ner']]

    indexed_word_score_tuple_list = []
    for i in range(len(word_score_tuple_list)):
        word, score = word_score_tuple_list[i]
        new_tuple = (word, score, i, masked_vector[i])
        indexed_word_score_tuple_list.append(new_tuple)

    # Sort by importance score
    if(sampling_strategy == 'topK'):
        indexed_word_score_tuple_list.sort(key=lambda x: x[1], reverse = True)
    elif(sampling_strategy == 'bottomK'):
        indexed_word_score_tuple_list.sort(key=lambda x: x[1], reverse = False)
    elif(sampling_strategy == 'randomK'):
        random.shuffle(indexed_word_score_tuple_list)

    # Creating a synonym map for num of terms to replace
    # It can handle cases where a word has no synonym - considers the next highest word
    synonym_map = {}
    for tup in indexed_word_score_tuple_list:
        if(len(synonym_map) > num_terms_to_replace):
            break
        else:
            # Only replaces unique term if same terms with different scores exists and is masked do not get synonym
            if(_wordvec_list_syns(tup[0]) != '' and tup[0] not in synonym_map and masked_vector[tup[2]] != 1):
                synonym_map[tup[0]] = _wordvec_list_syns(tup[0], 3 * num_output_samples)

    set_of_output_samples = set()
    i = j = 0 # Counter to keep a track of length of the word_score_tuple_list
    # j = 0 # Counter to keep a track of the number of terms replaced
    # Checks if both the required number of terms are replaced and required number of unique samples are generated
    new_word_index_tuple_list = []

    while(i < len(indexed_word_score_tuple_list) and len(set_of_output_samples) < num_output_samples):
        if(i == 0):
            copy_indexed_word_score_tuple_list = indexed_word_score_tuple_list.copy()
        word, score, index, masked_value  = indexed_word_score_tuple_list[i]

        # Checking the masked_one_hot_vector
        if(masked_vector[index] != 1 and j < num_terms_to_replace and word in synonym_map):
            new_tuple =  (synonym_map[word][random.choice(range(len(synonym_map[word])))],score, index, masked_value)
            copy_indexed_word_score_tuple_list[i] = (new_tuple)
            j += 1
        i += 1

        if(i == len(indexed_word_score_tuple_list) and len(set_of_output_samples) < num_output_samples):
            copy_indexed_word_score_tuple_list.sort(key=lambda x: x[2], reverse = False)
            new_string = ' '.join(tup[0] for tup in copy_indexed_word_score_tuple_list)
            set_of_output_samples.add(new_string)
            j = 0
            i = 0

        #print(set_of_output_samples)
    return set_of_output_samples

if __name__ == '__main__':
    # Loading Important Score Dictionary
    with open('/home/abijith/Downloads/SQuAD_v1.1_dev.pickle', mode='rb') as file:
        important_score_dict = pickle.load(file)
    K = 3
    id = 0
    sent  = 'How many g3p molecules leave in the cycle ?'
    word_score_tuple_list = important_score_dict[id]
    sample_set = replace_with_synonyms(sent, word_score_tuple_list, K, sampling_strategy = 'topK', num_output_samples = 5)
    print(word_score_tuple_list)
    print(sample_set)