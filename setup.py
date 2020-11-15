import shutil
import subprocess
import re
from pathlib import Path
from setuptools import setup, find_packages 

def _check_java():
    version = subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT).decode('utf-8')
    version = re.findall('"([^"]*)"', version)[0]
    version = version.split('.')[:2]
    if not version or not version == ['1', '8']:
        return False
    else:
        return True


def _get_requirements():
    """
    Here we check the local Java version for compatibility with pyspark/spark-nlp.
    If incompatible, we skip the pyspark/spark-nlp installs, which deactivates
    entity-aware features at runtime.
    """
    java_compat = _check_java()
    installs = [
        'config',
        'nltk',
        'numpy>=1.14.0,<1.18.0',
        'stop-words',
        'torch>=1.5.1',
        'transformers==3.1.0',
        'prefect==0.13.4',
        'pendulum==2.0.5',
        'dataclasses==0.6',
    ]
    if java_compat:
        installs += [
            'pyspark==2.4.0',
            'spark-nlp==2.5.2',
        ]
    return installs
        
 
setup(
    name='kitanaqa',
    url='https://github.com/searchableai/KitanaQA',
    author='Senzeyu Zhang, Mostafa Mirshekari, Aaron Sisto',
    author_email='aaron@searchable.ai',
    version='0.1.0',
    package_dir={"":"src"},
    python_requires=">=3.7.0",
    packages=find_packages("src"),
    install_requires=_get_requirements(),
    extras_require={
        'dev': [
            'pytest',
            'pytest-pep8',
            'pytest-cov',
        ]
    },
    description='Adversarial Training and Data Augmentation for Neural Question-Answering Models',
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP BERT adversarial training data augmentation deep learning pytorch google",
    license="Apache",
    package_data={
        'kitanaqa':['support/*.txt', 'support/*.zip','support/*.json']
    },
)
