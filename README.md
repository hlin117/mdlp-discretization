# Minimum Description Length Binning

This is an implementation of Usama Fayyad's entropy based
expert binning method.

Please read the original paper
<a href="http://sci2s.ugr.es/keel/pdf/algorithm/congreso/fayyad1993.pdf">here</a>
for more information.

# Installation and Usage

Install using pip 

```
pip install .
```


# Example

```
>>> from discretization import MDLP
>>> from sklearn.datasets import load_iris
>>> iris = load_iris()
>>> X = iris.data
>>> y = iris.target
>>> mdlp = MDLP()
>>> conv_X = mdlp.fit_transform(X, y)
```

As with all python packages, it is recommended to create a virtual environment
when using this project.

# Tests

To run the unit tests, install in development mode

```
pip install -e .
```

and then

```
py.test
```

# Development

To submit changes to this project, make sure that you have Cython installed and
submit the compiled *.cpp file along with changes to python code after running
installation locally.
