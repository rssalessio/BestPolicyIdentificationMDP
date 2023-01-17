from setuptools import setup, find_packages
from os import path


setup(name = 'BestPolicyIdentificationMDP',
    packages=find_packages(),
    version = '0.1.0',
    description = 'Python library for Best Policy Identification',
    url = 'https://github.com/rssalessio/BestPolicyIdentificationMDP',
    author = 'Alessio Russo',
    author_email = 'alessior@kth.se',
    install_requires=['numpy', 'scipy', 'cvxpy', 'jax'],
    license='MIT',
    zip_safe=False,
    python_requires='>=3.9',
)