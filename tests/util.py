import csv
import py
import numpy as np
from mdlp.discretization import MDLP


path_to_result = str(py.path.local(__file__).dirpath() / 'iris_result.csv')


def update_iris_test_result():
  from sklearn.datasets import load_iris
  iris = load_iris()
  new_result = MDLP(shuffle=False).fit_transform(iris.data, iris.target)
  with open(path_to_result, 'w') as outfile:
    csv.writer(outfile).writerows(new_result)

def load_iris_test_result():
  with open(path_to_result) as infile:
    reader = csv.reader(infile)
    return np.array(list(reader), float)
