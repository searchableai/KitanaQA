import pytest
import unittest
from doggmentator.augment.term_replacement import validate_inputs, get_scores, ReplaceTerms, DropTerms, RepeatTerms
from doggmentator.augment.generators import BaseGenerator, MisspReplace, SynonymReplace, _wordnet_syns
from doggmentator import get_logger
# init logging
logger = get_logger()

class TestTermReplacement(unittest.TestCase):
    def test_valid_inputs(self):
        num_replacements = 0
        num_output_sentences = 0
        sampling_strategy = 'topK'
        expected_result_1 = [1,1,'topK']
        result_1 = validate_inputs(num_replacements, num_output_sentences, sampling_strategy)

        num_replacements = 11
        num_output_sentences = 11
        expected_result_2 = [10, 10, 'random']
        sampling_strategy = 'random'
        result_2 = validate_inputs(num_replacements, num_output_sentences, sampling_strategy)

        num_replacements = 3
        num_output_sentences = 4
        sampling_strategy = 'bottomK'
        expected_result_3 = [3, 4, 'bottomK']
        result_3 = validate_inputs(num_replacements, num_output_sentences, sampling_strategy)

        assert isinstance(result_1, list)
        assert expected_result_1 == result_1
        assert isinstance(result_2, list)
        assert expected_result_2 == result_2
        assert isinstance(result_3, list)
        assert expected_result_3 == result_3

    def test_get_scores(self):
        original_sentence = 'what developmental network was discontinued after the shutdown of abc1 ?'
        expected_scores = [
                ('what', 0.0),
                ('developmental', 0.167),
                ('network', 0.167),
                ('was', 0.0),
                ('discontinued', 0.167),
                ('after', 0.0),
                ('the', 0.0),
                ('shutdown', 0.167),
                ('of', 0.0),
                ('abc1', 0.167),
                ('?', 0.167),
            ]
        results = get_scores(original_sentence.split())
        assert isinstance(results, list)
        assert all([any([pytest.approx(x[1], y[1]) for y in expected_scores]) for x in results])
        assert all([any([x[0] == y[0] for y in expected_scores]) for x in results])

    def test_get_entities(self):
        original_sentence = 'what developmental network was discontinued after the shutdown of abc1?'
        get_entity = ReplaceTerms()
        mask, tokens  = get_entity._get_entities(original_sentence)
        expected_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        expected_tokens = ['what', 'developmental', 'network', 'was', 'discontinued', 'after', 'the', 'shutdown', 'of', 'abc1', '?']
        assert len(mask) == len(tokens)
        assert isinstance(mask, list)
        assert isinstance(tokens, list)
        assert mask == expected_mask
        assert expected_tokens == tokens

    def test_replace_terms_synonym(self):
        original_sentence = 'what developmental network was discontinued after the shutdown of abc1?'
        importance_scores = [
                ('what', 0.0),
                ('developmental', 0.167),
                ('network', 0.167),
                ('was', 0.0),
                ('discontinued', 0.167),
                ('after', 0.0),
                ('the', 0.0),
                ('shutdown', 0.167),
                ('of', 0.0),
                ('abc1', 0.167),
                ('?', 0.167)
            ]

        syn_gen = ReplaceTerms(rep_type = 'synonym')
        syn_sentences = syn_gen.replace_terms(original_sentence, importance_scores, num_replacements=3, num_output_sents=1)
        assert isinstance(syn_sentences, list)
        assert len(syn_sentences) == 1

    def test_replace_terms_misspelling(self):
        original_sentence = 'The sky is absolutely beautiful in the summer'
        importance_scores = [
                ('The', 0.0),
                ('sky', 0.167),
                ('is', 0.0),
                ('absolutely', 0.2),
                ('beautiful', 0.167),
                ('in', 0.0),
                ('the', 0.0),
                ('summer', 0.155)
            ]
        misspellings = ReplaceTerms(rep_type = 'misspelling')
        missp_sentences = misspellings.replace_terms(original_sentence, importance_scores, num_replacements=2, num_output_sents=1)
        assert isinstance(missp_sentences, list)
        assert len(missp_sentences) == 1

    def test_dropwords(self):
        drop_word_sents = DropTerms()
        original_sentence = "Andy's friend just ate an apple and a bananna?!"
        dropped_sentences = drop_word_sents.drop_terms(sentence=original_sentence, num_terms=2, num_output_sents=2)
        assert isinstance(dropped_sentences, list)
        assert len(dropped_sentences) == 2

    def test_replaceterms(self):
        repeat_word_sents = RepeatTerms()
        original_sentence = "I am mr robot"
        repeated_sentences = repeat_word_sents.repeat_terms(sentence=original_sentence, num_terms=1, num_output_sents=1)
        assert isinstance(repeated_sentences, list)
        assert len(repeated_sentences) == 1
        assert repeated_sentences[0] == 'I am am mr robot'


if __name__ == '__main__':
    pass
