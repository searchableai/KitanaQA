<h1 align="center">Doggmentator</h1>
<p align="center">Adversarial Training and Data Augmentation for Question-Answer Models</p>

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

Doggmentator is an adversarial training and data augmentation framework for fine-tuning NLP language models on question-answering datasets

### *Why Doggmentator?*
While NLP models have made incredible progress on curated question-answer datasets in recent years, they are still brittle and unpredictable in production environments, making productization and enterprise adoption problematic. Doggmentator provides resources to "robustify" Transformer-based question-answer models against many types of natural and synthetic noise. The major features are:
1. **Adversarial Training** can increase both robustness and performance of fine-tuned Transformer QA models. Here, we implement an embedding-space perturbation method to simulate synthetic noise in model inputs. Comparisons to baselines like BERT-base show remarkable performance gains:

Model | em | f1
--- | --- | ---
BERT-base | 80 | 88.5
BERT-base (ALUM) | 82 | 88.99

2. **Augment your dataset** to increase model generalizability and robustness using token-level perturbations. While Adversarial Training provides some measure of robustness against bounded perturbations, Augmentation can accomodate a wide range of naturally-occuring noise in user input. We provide tools to augment existing SQuAD-like datasets by perturbing the examples along a number of different dimensions, including synonym replacement, misspelling, and deletion.

3. **Workflow Automation** to prototype robust NLP models faster for research and production. This package is structured for extremely easy use and deployment. Using Prefect Flows, training, evaluation, and model selection can be executed in a single line of code, enabling faster iteration and easier itergration of research into production pipelines.

# Features

## Adversarial Training
Our implementation is based on the ALUM model, proposed here [https://arxiv.org/pdf/2004.08994.pdf]. We have corrected a number of issues in the original formalism and improved algorithm robustness, adding flexible adversarial hyperparameter scheduling, as well as support for fp16.

## Augmenting Input Data
The following perturbation methods are available to augment SQuAD-like data:
- Synonym Replacement (SR) via 1) constrained word2vec [https://arxiv.org/pdf/1603.00892.pdf], and 2) MLM using BERT
```diff
- (original)  How many species of plants were *[recorded]* in Egypt?
+ (augmented) How many species of plants were *[registered]* in Egypt?
```
- Random Deletion (RD) using entity-aware term selection
```diff
- (original)  How many species of plants *[were]* recorded in Egypt?
+ (augmented) How many species of plants *[]* registered in Egypt?
```
- Random Misspelling (RM) using open-source common misspellings datasets
```diff
- (original)  How *[many]* species of plants were recorded in Egypt?
+ (augmented) How *[mony]* species of plants were registered in Egypt?
```
Each perturbation type also supports custom term importance sampling, e.g. as generated using a MLM
```diff
- (How, 0.179), (many, 0.254), (species, 0.123), (of, 0.03), (plants, 0.136) (were, 0.039), (recorded, 0.067), (in, 0.012), (Egypt, 0.159)
```
Perturbation types can also be flexibly applied in combination with different frequencies for fine-grained control of natural noise profiles
```diff
- (original)  How *[many]* species *[of]* plants *[were]* recorded in Egypt?
+ (augmented) How *[mony]* species *[]* plants *[]* recorded in Egypt?
```
## ML Flows
Using the Prefect library, Doggmenetator makes it increadibly easy to combine different workflows for end-to-end training/evaluation/model selection. This system also supports rapid iteration in hyperparameter search by easily specifying each experimental condition and deploying independently. You can even get training results reported directly in Slack!!!

# Installation
- Our entity-aware data augmentations make use of the John Snow Labs spark-nlp library, which requires pyspark. Ensure Java v8 is set by default for pyspark compatibility:
- ```sudo apt install openjdk-8-jdk```
- ```sudo update-alternatives --config java```
- ```java -version```

- Word2Vec-based synonym replacement relies on pretrained embeddings found in the /support directory.
Unzip word vec files in /support directory

- Install the package
```python setup.py install```

# Getting Started
```python run_pipeline.py --args=args.json```

# More Examples
- run_pipeline (Regular, ALUM, train + evaluate)
- augment_squad

## Models Supported
We make use of the following models and their respective tokenizers and configurations provided by HuggingFace Inc.
- ALBERT
- BERT
- DistilBERT

## Contributing to Doggmentator

We welcome suggestions and contributions! Submit an issue or pull request and we will do our best to respond in a timely manner.
See [CONTRIBUTING.md](https://github.com/searchableai/Doggmentator/blob/master/CONTRIBUTING.md) for detailed information on contributing.

## Thanks!
- Huggingface Inc.
- John Snow Labs
- Torch community
