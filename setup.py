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
        'nltk',
        'numpy',
        'stop-words',
        'pyspark==2.4.0',
        'spark-nlp==2.5.2',
        'torch==1.3.1',
        'transformers==3.1.0',
        'prefect==0.13.4',
        'pendulum==2.0.5',
        'dataclasses==0.6',
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
