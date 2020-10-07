# Load SQuAD Dataset
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor, squad_convert_examples_to_features

class AugSquadTester:
    def __init__(
            self,
            num_replacements=1,
            sample_ratio=0.001,
            sampling_strategy='topK',
            sampling_k=1,
            is_training=True,
            out_prefix=None,
            save_freq=1000,
            from_checkpoint=False,
            verbose=False):
        self.num_replacements=num_replacements
        self.sample_ratio=sample_ratio
        self.sampling_k=sampling_k
        self.is_training = is_training
        self.out_prefix=out_prefix
        self.save_freq=save_freq
        self.from_checkpoint=from_checkpoint
        self.verbose=verbose

        # Load test data
        data_file = pkg_resources.resource_filename(
            'doggmentator', 'support/train-v1.1.json')
        with open(data_file, 'r') as f:
            data = json.load(f)

        hparams = { 
            "num_replacements": self.num_replacements,
            "sample_ratio":     self.sample_ratio,
            "sampling_strategy":self.sampling_strategy,
            "sampling_k":       self.sampling_k,
            "is_training":      self.is_training,
            "out_prefix":       self.out_prefix,
            "save_freq":        self.save_freq,
            "from_checkpoint":  self.from_checkpoint,
        }
        self.generator = SQuADDataset(data, **hparams)

    def load_and_check_data(self):
        self.generator._load_raw_data()
        assert isinstance(self.examples, List)
        assert len(self.examples) == 87719

    def generate_and_check_results(self):
        self.generator.generate()
        assert isinstance(self.dataset, List)
        assert len(generator.dataset) == generator.num_aug_examples

class AugSquadTest:
    hparams = {
        "num_replacements": 1,
        "sample_ratio":     0.001,
        "sampling_strategy":'topK',
        "sampling_k":       1,
        "is_training":      True,
        "out_prefix":       None,
        "save_freq":        1000,
        "from_checkpoint":  False,
    }
    
    def load_data(self):
        self.model_tester = AugSquadTester(**hparams)
        self.model_test.load_and_check_data()

    def generate(self):
        self.generator.generate_and_check_results()
