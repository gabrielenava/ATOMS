from setuptools import setup, find_packages

setup(
    name='atoms',
    version='0.2',
    packages=find_packages(),
    install_requires=[
        'osqp',
        'numpy',
        'matplotlib',
        'scipy',
    ],
)
