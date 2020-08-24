<h1 align="center">Doggmentator</h1>
<p align="center">Adversarial Training and Data Augmentation for NLP Models</p>

<p align="center">
  <a href="">[Doggmentator Documentation on ReadTheDocs]</a> 
  <br> <br>
  <a href="#about">About</a> •
  <a href="#setup">Setup</a> •
  <a href="#usage">Usage</a> •
  <a href="#design">Design</a> 
  <br> <br>
</p>

[![CircleCI](https://circleci.com/gh/searchableai/Doggmentator.svg?style=shield&circle-token=de6470b621d1b07e54466dd087b85b80bcedf36c)](https://github.com/searchableai/Doggmentator)

## About
Doggmentator is a Python framework for adversarial training and data augmentation for pre-trained NLP language model training.

## Augmenting Input Data
We try incorportating the following methods to augment original data following the research - 
[https://arxiv.org/pdf/1603.00892.pdf]
- Synonym Replacement (SR) - Replace with synonyms obtained from Wordnet using a variable similarity metric
- Random Insertion (RI) - Insert random tokens/sub-tokens within a token to mimic human typos, sampled to retain the natural distribution of mis-spellings
- Random Swap (RS) - Swap sub tokens randomly, another way to introduce natural and frequently occuring typing errors
- Random Deletion (RD) - Delete token/sub-tokens randomly to represent missed out tokens while typing
- Random Misspelling (RM) - Introduce mis-spellings randomly sampled from a long list of common mis-spellings for a given token
- Query Reformulation (QR) - Train the model with different possible versions of the query, to increase the generalizability and robustness of the model


### *Why Doggmentator?*
There are lots of reasons which motivated us to work on this project:
1. **Understand NLP models better** by using adversarial training and data augmentation understand the effects of it on model generalizability and robustness
2. **Create a general framework** to automate and prototype different NLP models faster for research and production
3. **Augment your dataset** to increase model generalization and robustness downstream

### Installation
- Ensure Java v8 is set by default for pyspark compatibility:
```sudo apt install openjdk-8-jdk```
```sudo update-alternatives --config java```
```java -version```

- Setup word vectors
Unzip word vec files in /support directory

- Install the package
```python setup.py install```

## Usage
- To run the file
```python run_squad_hf_adv_aug_full.py```

Note:
1.) Change the model configuration using the argument parameter(args)
2.) Please have checkpoint-squad.json and train-v1.1.json to run from a checkpoint
3.) Make changes to the cache directory path in args

## Models Used
We make use of the following models and their respective tokenizers and configurations provided by HuggingFace Inc.
- ALBERT
- BERT
- DistilBERT

## Contributing to Doggmentator

We welcome suggestions and contributions! Submit an issue or pull request and we will do our best to respond in a timely manner.
See [CONTRIBUTING.md](https://github.com/searchableai/Doggmentator/blob/master/CONTRIBUTING.md) for detailed information on contributing.
