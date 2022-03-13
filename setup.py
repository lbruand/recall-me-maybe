#!/usr/bin/python
# -*-coding: utf-8 -*-

from os import path

from setuptools import setup

here = path.abspath(path.dirname(__file__))

with open('README_pypi.rst') as f:
    long_description = f.read()

setup(
    name="recallme",
    description="Recall me maybe is an open source, MIT-licensed Python package for plotting recall precision waffle charts from a confusion matrix.",
    keywords="matplotlib waffle chart pie plot data visualization precision recall confusion matrix",
    long_description=long_description,
    license='MIT',
    author="Lucas Bruand",
    author_email="l.bruand.pro@gmail.com",
    url="https://github.com/lbruand/recall-me-maybe",
    packages=['recallme'],
    install_requires=['matplotlib', 'numpy'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Visualization',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
    package_data={
        'recallme': []
    },
    include_package_data=True
)

