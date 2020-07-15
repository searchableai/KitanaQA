import pytest
from doggmentator import term_replacement
from term_replacement import validate_inputs, get_scores

class TestTermReplacement():

    def test_validate_inputs(self):
        from doggmentator.term_replacement import validate_inputs
        num_replacements = 0
        num_output_sentences = 0
        sampling_strategy = 'topK'
        expected_result_1 = [1,1,'topK']
        result_1 = validate_inputs(num_replacements, num_output_sentences, sampling_strategy)
        num_replacements = 11
        num_output_sentences = 11
        expected_result_2 = [10, 10, 'random']
        # Removed Sampling Strategy - Watch out
        result_2 = validate_inputs(num_replacements, num_output_sentences)
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
        from doggmentator.term_replacement import get_scores
        original_sentence = 'what developmental network was discontinued after the shutdown of abc1 ?'
        expected_scores = [('what', 13.653033256530762), ('developmental', 258.72607421875), ('network', 157.8809356689453), ('was', 18.151954650878906), ('discontinued', 30.241737365722656), ('after', 70.61669921875), ('the', 4.491329193115234), ('shutdown', 32.54951477050781), ('of', 11.050531387329102), ('abc1', 54.5350456237793), ('?', 0)]
        result = get_scores(original_sentence.split())
        assert isinstance(result, list)
        print('expected_scores = ', expected_scores)
        print('resultant scores  = ', result)
        assert expected_scores == result

if __name__ == '__main__':
    run = TestTermReplacement()
    run.test_get_scores()
    run.test_validate_inputs()