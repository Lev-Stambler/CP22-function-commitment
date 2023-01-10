from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

import os

def main():

      setup(name='CP22',
             version='1.0.1',
             description='CP22',
             ext_modules=[Extension("intmm", ["int_mat_mul.c"])])
                  # [ext], compiler_directives={'language_level' : "3"}))
if (__name__ == "__main__"):
  main()