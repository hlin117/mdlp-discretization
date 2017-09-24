# Minimum Description Length Binning

This is an implementation of Usama Fayyad's entropy based
expert binning method.

Please read the original paper
<a href="http://sci2s.ugr.es/keel/pdf/algorithm/congreso/fayyad1993.pdf">here</a>
for more information.

# Installation and Usage

You'll need to install Cython and numpy first, if you haven't already.

This project is not yet on PyPI, but you can install the lastest `master` through pip like so:

```
$ pip install https://github.com/hlin117/mdlp-discretization/archive/master.zip
```

Here's a quick usage example:

```
>>> from discretization import MDLP
>>> from sklearn.datasets import load_iris
>>> iris = load_iris()
>>> X = iris.data
>>> y = iris.target
>>> mdlp = MDLP()
>>> X_discrete = mdlp.fit_transform(X, y)
```

Afterwards, `X_discrete` will have the same shape as X but will be integer valued.

# Tests

To run the unit tests, make sure you have `nose` installed. Afterwards,

```
$ make test
```

should do the trick.
