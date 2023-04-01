from setuptools import setup, find_packages

setup(
    name='atoms',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'osqp',
        'numpy',
        'matplotlib',
        'scipy',
    ],
)
