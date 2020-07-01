from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import nltk
import itertools

nltk.download('stopwords')
nltk.download('punkt')


# randomly drop K stop words in sentence, return N variations
def dropwords(sentence, N, K=None):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sentence)
    stop_words_indices = []
    for idx, word in enumerate(word_tokens):
        if word in stop_words:
            stop_words_indices.append(idx)
    dropped_sentences = []
    if len(stop_words_indices) == 0:
        return dropped_sentences

    # Randomly choose K indices from all stop_words_indices
    # If K is not specified, generate all possible combinations
    comb = []
    if not K:
        for r in range(1, len(stop_words_indices) + 1):
            comb += list(itertools.combinations(stop_words_indices, r))
    else:
        K = min(len(stop_words_indices), max(1, K))
        comb += list(itertools.combinations(stop_words_indices, K))

    N_chosen_indices = [comb[idx] for idx in np.random.choice(len(comb), size=min(N, len(comb)), replace=False)]
    for chosen_indices in N_chosen_indices:
        new_words = [word_tokens[idx] for idx in range(len(word_tokens)) if idx not in chosen_indices]
        dropped_sentence = ' '.join(new_words)
        dropped_sentences.append(dropped_sentence)
    return dropped_sentences

if __name__ == '__main__':
    raw_sentence = "Andy's friend just ate an apple and a banana?!"
    dropped_sentences = dropwords(raw_sentence, N=2, K=None)
    print(raw_sentence)
    print(dropped_sentences)
    print(tokenizer.tokenize(raw_sentence))
    for dropped_sentence in dropped_sentences:
        print(tokenizer.tokenize(dropped_sentence))

    dropped_sentences = dropwords(raw_sentence, N=2, K=2)
    print(raw_sentence)
    print(dropped_sentences)
    print(tokenizer.tokenize(raw_sentence))
    for dropped_sentence in dropped_sentences:
        print(tokenizer.tokenize(dropped_sentence))