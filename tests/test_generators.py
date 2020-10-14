import pytest
import unittest
from doggmentator.augment.term_replacement import validate_inputs, get_scores, ReplaceTerms, DropTerms
from doggmentator.augment.generators import BaseGenerator, MisspReplace, SynonymReplace, MLMSynonymReplace

class TestGenerators(unittest.TestCase):
    def test_valid_input(self):
        base_gen = BaseGenerator()
        test_str  = 'What is! 12.32% my, \n name'
        sanitized_str = base_gen._check_sent(test_str)
        expected_string = 'what is 12.32 my name'
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

    def test_misspelling_generator(self):
        missp_gen = MisspReplace()
        misspellings = missp_gen.generate('apple', 5)
        assert isinstance(misspellings, list)
        assert all([isinstance(x, str) for x in misspellings])
        assert len(misspellings) == 5

    def test_w2v_synonym_generator(self):
        syn_gen = SynonymReplace()
        synonyms = syn_gen.generate('apple', 3, **{'similarity_thre': 0.75})
        assert isinstance(synonyms, list)
        assert all([isinstance(x, str) for x in synonyms])
        assert len(synonyms) == 3

    def test_mlm_synonym_generator(self):
        syn_gen = MLMSynonymReplace()
        sent = 'I was born in a small town'
        synonyms = syn_gen.generate('small', 3, **{'toks': sent.split(), 'token_idx': 5})
        assert isinstance(synonyms, list)
        assert all([isinstance(x, str) for x in synonyms])
        assert len(synonyms) == 3

    def test_load_misspellings(self):
        missp_gen = MisspReplace()
        missp_gen._load_misspellings()
        assert isinstance(missp_gen._missp, dict)

    def test_load_w2v_embeds(self):
        syn_gen = SynonymReplace()
        syn_gen._load_embeddings()
        assert isinstance(syn_gen._vecs, dict)


if __name__ == '__main__':
    pass
