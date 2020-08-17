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

[*** Make a gif of a command line working example and add it in here ***]

## About

Doggmentator is a Python framework for adversarial training and data augmentation for pre-trained NLP language model training.

> If you're looking for information about Doggmentators menagerie of pre-trained models, please find it here.
[** Add link and description to model details **]

## Augmenting Input Data
We try incorportating the following methods to augment original data following the research - 
[https://arxiv.org/pdf/1603.00892.pdf]
- Synonym Replacement (SR) 
- Random Insertion (RI)
- Random Swap (RS)
- Random Deletion (RD)
- Random Misspelling (RM)
- Query Reformulation (QR)

## Github/Stack overflow/Slack Channel

### *Why Doggmentator?*
There are lots of reasons which motivated us to work on this project:
1. **Understand NLP models better** by using adversarial training and data augmentation understand the effects of it on model generalizability and robustness
2. **Create a general framework** to automate and prototype different NLP models faster for research and production
3. **Augment your dataset** to increase model generalization and robustness downstream

## Setup

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
- [** Add usage instructions **]

### Examples

### Augmenting Text

The 'embedding' augmentation recipe uses counter-fitted embedding nearest-neighbors to augment data.

### Training Models

[*** Planning to add a sepearte readme going into a description of all the models we tried and their performance ****]

#### Training Examples
[*** Add in training examples which shows the capabilities and the limitaions of our model ****]

## Design
[*** Talk about our basic system design with a block diagram, makes it intuitive to understand the concept ***]

## Contributing to Doggmentator

We welcome suggestions and contributions! Submit an issue or pull request and we will do our best to respond in a timely manner.
See [CONTRIBUTING.md](https://github.com/searchableai/Doggmentator/blob/master/CONTRIBUTING.md) for detailed information on contributing.
