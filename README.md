<p align="center"><img src="/assets/img/searchable-logo_full-lockup-horizontal_dark.png" width="460"></p>
&nbsp
<h1 align="center">KatanaQA</h1>
<p align="center"><b>[A]dversarial [T]raining [AN]d [A]ugmentation for [Q]uestion-[A]nswering</b></p>
<p align="center">
  <a href="#about">About</a> •
  <a href="#features">Features</a> •
  <a href="#installation">Install</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#examples">Examples</a>
  <br> <br>
</p>

[![CircleCI](https://circleci.com/gh/searchableai/KatanaQA.svg?style=shield&circle-token=de6470b621d1b07e54466dd087b85b80bcedf36c)](https://github.com/searchableai/KatanaQA)

# About

KatanaQA is an adversarial training and data augmentation framework for fine-tuning Transformer-based language models on question-answering datasets


## *Why KatanaQA?*
While NLP models have made incredible progress on curated question-answer datasets in recent years, they are still brittle and unpredictable in production environments, making productization and enterprise adoption problematic. KatanaQA provides resources to "robustify" Transformer-based question-answer models against many types of natural and synthetic noise. The major features are:
1. **Adversarial Training** can increase both robustness and performance of fine-tuned Transformer QA models. Here, we implement *virtual adversarial training*, which introduces embedding-space perturbations during fine-tuning to encourage the model to produce more stable results in the presence of noisy inputs.

  Our experiments with BERT finetuned on the SQuAD v1.1 question answering dataset show a marked improvement in f1 and em scores:

  Model | em | f1
  --- | --- | ---
  BERT-base | 80.8 | 88.5
  **BERT-base (ALUM)** | **81.97** | **88.92**

2. **Augment Your Dataset** to increase model generalizability and robustness using token-level perturbations. While Adversarial Training provides some measure of robustness against bounded perturbations, Augmentation can accomodate a wide range of naturally-occuring noise in user input. We provide tools to augment existing SQuAD-like datasets by perturbing the examples along a number of different dimensions, including synonym replacement, misspelling, repetition and deletion.

3. **Workflow Automation** to prototype robust NLP models faster for research and production. This package is structured for extremely easy use and deployment. Using Prefect Flows, training, evaluation, and model selection can be executed in a single line of code, enabling faster iteration and easier itergration of research into production pipelines.

# Features

## Adversarial Training
Our implementation is based on the smoothness-inducing regularization approach proposed [here](https://arxiv.org/pdf/1605.07725.pdf). We have updated the implementation for fine-tuning on question-answer datasets, and added additional features like adversarial hyperparameter scheduling, and support for mixed-precision training.

## Adversarial Attack
A key measure of robustness in neural networks is the so-called white-box adversarial attack. In the context of Transformer-based Question-Answer models, this attack seeks to inject noise into the model's input embeddings and assess performance on the original labels. Here, we implement the projected gradient descent (PGD) attack mechanism, bounded by the norm-ball. Metrics can be calculated for non-adversarial and adversarial evaluation, making robustness studies more streamlined and accessible.

## Data Augmentation
The following perturbation methods are available to augment SQuAD-like data:
- Synonym Replacement (SR) via 1) constrained [word2vec](https://arxiv.org/pdf/1603.00892.pdf), and 2) MLM using BERT
```diff
- (original)  How many species of plants were *recorded* in Egypt?
+ (augmented) How many species of plants were *registered* in Egypt?
```
- Random Deletion (RD) using entity-aware term selection
```diff
- (original)  How many species of plants *were* recorded in Egypt?
+ (augmented) How many species of plants ** recorded in Egypt?
```
- Random Repetition (RR) using entity-aware term selection
```diff
- (original)  How many species of plants *were* recorded in Egypt?
+ (augmented) How many species of plants *were were* recorded in Egypt?
```
- Random Misspelling (RM) using open-source common misspellings datasets
    -- *sources: [wiki](https://en.wikipedia.org/wiki/Wikipedia:Lists_of_common_misspellings), [brikbeck](https://www.dcs.bbk.ac.uk/~ROGER/corpora.html)*
```diff
- (original)  How *many* species of plants were recorded in Egypt?
+ (augmented) How *mony* species of plants were recorded in Egypt?
```
Perturbation types can be flexibly applied in combination with different frequencies for fine-grained control of natural noise profiles
```diff
- (original)  How *many* species *of* plants *were* recorded in Egypt?
+ (augmented) How *mony* species ** plants ** recorded in Egypt?
```
Each perturbation type also supports custom term importance sampling, e.g. as generated using a MLM  
```(How, 0.179), (many, 0.254), (species, 0.123), (of, 0.03), (plants, 0.136) (were, 0.039), (recorded, 0.067), (in, 0.012), (Egypt, 0.159)```

## ML Flows
Using the Prefect library, KatanaQA makes it increadibly easy to combine different workflows for end-to-end training/evaluation/model selection. This system also supports rapid iteration in hyperparameter search by easily specifying each experimental condition and deploying independently. You can even get results [reported directly in Slack](https://docs.prefect.io/core/advanced_tutorials/slack-notifications.html)!!!

# Installation
Our entity-aware data augmentations make use of the John Snow Labs [spark-nlp](https://github.com/JohnSnowLabs/spark-nlp) library, which requires pyspark. To enable this feature, make sure Java v8 is set by default for pyspark compatibility:
- ```sudo apt install openjdk-8-jdk```
- ```sudo update-alternatives --config java```
- ```java -version```

Install the package
- ```python setup.py install```

# Getting Started
- ```python run_pipeline.py --args=args.json```

# Examples

## *Augmentation*
- [Generating Augmented Datasets](examples/augment_squad)
- [Custom Text Perturbations](examples/generate_token_perturbations)
- [MLM Importance Scores](examples/generate_importance_scores_with_mlm)

## *Training and Evaluation*
- [Automated Training Pipeline](examples/training_and_evaluation)
- [Adversarial Training](examples/alum_training_and_evaluation)
- [Adversarial Attack](examples/adversarial_attack)

## Models Supported
We make use of the following models and their respective tokenizers and configurations provided by HuggingFace Inc.
- ALBERT
- BERT
- DistilBERT

### Contributing to KatanaQA

We welcome suggestions and contributions! Submit an issue or pull request and we will do our best to respond in a timely manner.
See [CONTRIBUTING.md](https://github.com/searchableai/KatanaQA/blob/master/CONTRIBUTING.md) for detailed information on contributing.

### Thanks!
- John Snow Labs
- Hugging Face Inc.
- pytorch community
