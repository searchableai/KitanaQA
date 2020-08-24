Why Doggmentator?
^^^^^^^^^^^^^^^^^
| There are lots of reasons which motivated us to work on this project:
| 1. **Understand NLP models better** by using adversarial training and data augmentation understand the effects of it on model generalizability and robustness
| 2. **Create a general framework** to automate and prototype different NLP models faster for research and production
| 3. **Augment your dataset** to increase model generalization and robustness downstream

Augmenting Input Data
^^^^^^^^^^^^^^^^^^^^^
| We try incorporating the following methods to augment original data following the research - [https://arxiv.org/pdf/1603.00892.pdf]
- Synonym Replacement (SR) - Replace with synonyms obtained from Wordnet using a variable similarity metric
- Random Insertion (RI) - Insert random tokens/sub-tokens within a token to mimic human typos, sampled to retain the natural distribution of mis-spellings
- Random Swap (RS) - Swap sub tokens randomly, another way to introduce natural and frequently occuring typing errors
- Random Deletion (RD) - Delete token/sub-tokens randomly to represent missed out tokens while typing
- Random Misspelling (RM) - Introduce mis-spellings randomly sampled from a long list of common mis-spellings for a given token
- Query Reformulation (QR) - Train the model with different possible versions of the query, to increase the generalizability and robustness of the model
