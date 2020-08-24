Setup
^^^^^
| - Ensure Java v8 is set by default for pyspark compatibility:
| ``sudo apt install openjdk-8-jdk``
| ``sudo update-alternatives --config java``
| ``java -version``

| - Setup word vectors
| Unzip word vec files in /support directory

| - Install the package
| ``python setup.py install``

Usage
^^^^^
| - To run the file
python run_squad_hf_adv_aug_full.py

Note:
1.) Change the model configuration using the argument parameter(args)
2.) Please have checkpoint-squad.json and train-v1.1.json to run from a checkpoint
3.) Make changes to the cache directory path in args