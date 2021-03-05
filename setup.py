#!/usr/bin/env python3
# Inspired from https://github.com/kennethreitz/setup.py

from pathlib import Path
from setuptools import setup, find_packages

NAME = 'wavebeat'
DESCRIPTION = 'End-to-end beating tracking with temporal convolutional networks'
URL = 'https://github.com/csteinmetz1/wavebeat'
EMAIL = 'c.j.steinmetz@qmul.ac.uk'
AUTHOR = 'Christian Steinmetz'
REQUIRES_PYTHON = '>=3.8.0'
VERSION = "0.0.1"

HERE = Path(__file__).parent

try:
    with open(HERE / "README.md", encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=['wavebeat'],
    install_requires=['torch==1.7.1',
                      'torchaudio==0.7.2', 
                      'pytorch_lightning==1.1.8',
                      'torchvision==0.8.2',
                      'scipy', 
                      'numpy',
                      'cython', 
                      'mir_eval', 
                      'madmom', 
                      'Pillow', 
                      'julius', 
                      'matplotlib',
                      'tqdm',
                      'soxbindings'],
    extras_require={"test": ['resampy']},
    include_package_data=True,
    license=' GPL-3.0 License',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Multimedia :: Sound/Audio',
        'Topic :: Scientific/Engineering',
    ],
)