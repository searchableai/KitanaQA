import shutil
from pathlib import Path
from setuptools import setup, find_packages 
 
setup(
    name='Doggmentator',
    url='https://github.com/searchableai/doggmentator',
    author='Senzeyu Zhang, Abijith Vijayendra, Aaron Sisto',
    author_email='aaron@searchable.ai',
    version='0.1',
    package_dir={"":"src"},
    python_requires=">=3.5.0",
    packages=find_packages("src"),
    install_requires=[
        'config',
        'numpy',
        'nltk',
        'stop-words',
        'transformers==3.1.0',
        'pyspark==2.4.0',
        'spark-nlp==2.5.2'
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-pep8',
            'pytest-cov',
        ]
    },
    description='Data augmentation for language models',
    package_data={
        'doggmentator':['support/*.txt','support/*.json']
    },
)
