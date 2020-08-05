Doggmentator

[![CircleCI](https://circleci.com/gh/searchableai/Doggmentator.svg?style=svg&circle-token=de6470b621d1b07e54466dd087b85b80bcedf36c)](https://github.com/searchableai/Doggmentator)


- Synonym Replacement (SR) https://arxiv.org/pdf/1603.00892.pdf
- Random Insertion (RI)
- Random Swap (RS)
- Random Deletion (RD)
- Random Misspelling (RM)
- Query Reformulation (QR)


Install Instructions:
- Ensure Java v8 is set by default for pyspark compatibility:
```sudo apt install openjdk-8-jdk```
```sudo update-alternatives --config java```
```java -version```

- Setup word vectors
Unzip word vec files in /support directory

- Install the package
```python setup.py install```


