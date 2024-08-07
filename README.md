# ATOMS

[![Python Test](https://github.com/gabrielenava/ATOMS/actions/workflows/python-test.yml/badge.svg)](https://github.com/gabrielenava/ATOMS/actions/workflows/python-test.yml)

Advanced Tools for Multi-Body Systems

ATOMS is a Python package that collects useful tools to operate with systems of rigid bodies.

### List of dependencies

- [osqp](https://osqp.org/)
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/stable/index.html)

### Available classes

- [linearMPC](atoms/linearMPC.py): implements Model Predictive Control for linear systems using OSQP;
- [kalmanFilter](atoms/kalmanFilter.py): implementation of the Kalman Filter;
- [import_data](iNomaly/import_data.py): import, process, split and plot data in `.mat` format;
- [atoms_helpers](iNomaly/inomaly_helpers.py): helpers methods and logger to be used in the other classes of the package;
- [one_class_svm](iNomaly/one_class_svm.py): wrapper of the one class support vector machines (SVM) from scikit-learn.

### Examples

See also the [examples](examples) folder.

### Installation and usage

Tested on Ubuntu 20.04 LTS.

Firstly, `git clone` this repository. To install the dependencies in a venv virtual environment run:

```
python -m venv atomsenv
source ./atomsenv/bin/activate
```

then, run the command `python3 setup.py install`or `pip install -r requirements.txt` to install the required dependencies.

#### Maintainer

Gabriele Nava, [@gabrielenava](https://github.com/gabrielenava)
