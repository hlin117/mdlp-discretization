from distutils.core import setup

try:
    from Cython.Build import cythonize
    import numpy
except ImportError:
    raise RuntimeError('Unsatisfied setup dependencies, please install cython and numpy first')


setup(
    name='mdlp-discretization',
    version='0.1',
    description='Discretization of continuous features using Fayyad\'s MDLP stop criterion.',
    author='Henry Lin',
    url='https://github.com/hlin117/mdlp-discretization',
    install_requires=['numpy', 'scikit-learn', 'scipy'],
    setup_requires=['cython', 'numpy'],
    py_modules=['discretization'],
    ext_modules=cythonize('*.pyx', language='c++'),
    include_dirs=[numpy.get_include()]
)
