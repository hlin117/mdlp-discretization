# Minimum Description Length Binning

This is an implementation of Usama Fayyad's entropy based
expert binning method.

Please read the original paper
<a href="http://web.donga.ac.kr/kjunwoo/files/Multi%20interval%20discretization%20of%20continuous%20valued%20attributes%20for%20classification%20learning.pdf">here</a>
for more information.

# Installation and Usage

Install using pip

```
pip install git+https://github.com/hlin117/mdlp-discretization
```

As with all python packages, it is recommended to create a virtual environment
when using this project.

# Example

```
>>> from mdlp.discretization import MDLP
>>> from sklearn.datasets import load_iris
>>> transformer = MDLP()
>>> iris = load_iris()
>>> X, y = iris.data, iris.target
>>> X_disc = transformer.fit_transform(X, y)
```

# Tests

To run the unit tests, clone the repo and install in development mode

```
git clone https://github.com/hlin117/mdlp-discretization
cd mdlp-discretization
pip install -e .
```

then run tests with py.test

```
py.test tests
```

# Development

To submit changes to this project, make sure that you have Cython installed and
submit the compiled *.cpp file along with changes to python code after running
installation locally.
