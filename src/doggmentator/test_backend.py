import pytest
from doggmentator.term_replacement import validate_inputs, get_scores, ReplaceTerms, DropTerms
from doggmentator.generators import BaseGenerator, MisspReplace, SynonymReplace, _wordnet_syns
from doggmentator import get_logger
# init logging
logger = get_logger()

class TestTermReplacement():

    def test_validate_inputs(self):
        num_replacements = 0
        num_output_sentences = 0
        sampling_strategy = 'topK'
        expected_result_1 = [1,1,'topK']
        result_1 = validate_inputs(num_replacements, num_output_sentences, sampling_strategy)

        num_replacements = 11
        num_output_sentences = 11
        expected_result_2 = [10, 10, 'random']
        sampling_strategy = 'random'
        # Removed Sampling Strategy - Watch out
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
        expected_scores = [('what', 0.0), ('developmental', 0.16666666666666666), ('network', 0.16666666666666666), ('was', 0.0), ('discontinued', 0.16666666666666666), ('after', 0.0), ('the', 0.0), ('shutdown', 0.16666666666666666), ('of', 0.0), ('abc1', 0.16666666666666666), ('?', 0.16666666666666666)]
        result = get_scores(original_sentence.split())
        assert isinstance(result, list)
        assert expected_scores == result

class TestReplaceTerm():

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
        importance_scores = [('what', 0.0), ('developmental', 0.16666666666666666), ('network', 0.16666666666666666),
                           ('was', 0.0), ('discontinued', 0.16666666666666666), ('after', 0.0), ('the', 0.0),
                           ('shutdown', 0.16666666666666666), ('of', 0.0), ('abc1', 0.16666666666666666),
                           ('?', 0.16666666666666666)]

        syn_gen = ReplaceTerms(rep_type = 'synonym')
        syn_sentences = syn_gen.replace_terms(original_sentence, importance_scores, num_replacements=3, num_output_sents=1)
        assert isinstance(syn_sentences, list)

    def test_replace_terms_missplelling(self):
        original_sentence = 'The sky is absolutely beautiful in the summer'
        importance_scores = [('The', 0.0), ('sky', 0.16666666666666666), ('is', 0.0),
                                 ('absolutely', 0.2), ('beautiful', 0.16666666666666666), ('in', 0.0), ('the', 0.0),
                                 ('summer', 0.155)]

        misspellings = ReplaceTerms(rep_type = 'misspelling')
        mis_spelt_sentences = misspellings.replace_terms(original_sentence, importance_scores, num_replacements=2, sampling_k=3)
        assert isinstance(mis_spelt_sentences, list)

class TestGenerators():

    def test_check_sent(self):
        base_gen = BaseGenerator()
        test_str  = 'What is! 12.32% my, \n name'
        sanitized_str = base_gen._check_sent(test_str)
        expected_string = 'what is 12.32 my  name'
        assert isinstance(sanitized_str, str)
        assert expected_string == sanitized_str

    def test_cosine_similarity(self):
        a = [2, 0, 1, 1, 0, 2, 1, 1]
        b = [2, 1, 1, 0, 1, 1, 1, 1]
        base_gen = BaseGenerator()
        sim = base_gen._cosine_similarity(a, b)
        sim = sim.item()
        assert isinstance(sim, float)
        assert sim == pytest.approx(0.82, 0.02)

    # Need some change in the main script
    def test_wordnet_syns(self):
        sim = _wordnet_syns('apple', 2)
        assert isinstance(sim, str)

    def test_misspelling_generator(self):
        mis_gen = MisspReplace()
        misspellings =mis_gen.generate('apple', 5)
        assert isinstance(misspellings, list)
        assert len(misspellings) == 5

    def test_synonym_generator(self):
        syn_gen = SynonymReplace()
        synonyms =syn_gen.generate('apple', 3, 0.75)
        assert isinstance(synonyms, list)
        assert len(synonyms) == 3

class TestDropWords():

    def test_dropwords(self):
        drop_word = DropTerms()
        original_sentence = "Andy's friend just ate an apple and a bananna?!"
        dropped_sentences = drop_word.drop_terms(sentence=original_sentence, num_terms=2,num_output_sents=2)
        assert isinstance(dropped_sentences, list)
        assert len(dropped_sentences) == 2


if __name__ == '__main__':
    pass