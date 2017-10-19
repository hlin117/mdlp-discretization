from setuptools import Extension, find_packages, setup

if __name__ == '__main__':
  try:
    from Cython.setuptools import build_ext
  except:
    from setuptools.command.build_ext import build_ext
    sources = ['mdlp/_mdlp.pyx']
  else:
    sources = ['mdlp/_mdlp.cpp']

  class custom_build_ext(build_ext):
    def run(self):
      import numpy
      self.include_dirs.append(numpy.get_include())
      build_ext.run(self)

  cpp_ext = Extension(
      'mdlp._mdlp',
      sources=sources,
      libraries=[],
      include_dirs=[],
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
      cmdclass={'build_ext': custom_build_ext},
  )
