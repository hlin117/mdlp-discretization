import numpy as np
from setuptools import setup, Extension, find_packages

if __name__ == '__main__':
  cpp_ext = Extension(
      'mdlp._mdlp',
      sources=['mdlp/_mdlp.pyx'],
      libraries=[],
      include_dirs=[np.get_include()],
      language='c++',
  )

  setup(
      name='mdlp-discretization',
      version='0.1',
      description="""Implements the MDLP discretization criterion from Usama Fayyad's paper "Multi-Interval Discretization of Continuous-Valued Attributes for Classification Learning".""",
      license='BSD 3 Clause',
      url='github.com/hlin117/mdlp-discretization',
      author='Henry Lin',
      author_email='hlin117@gmail.com',
      install_requires=[
        'setuptools>=18.0',
        'cython>=0.23.2',
        'numpy>=1.11.2',
        'scipy>=0.18.1',
        'scikit-learn>=0.18.1',
        'pytest>=3.2.2',
      ],
      packages=find_packages(),
      ext_modules=[cpp_ext],
  )
