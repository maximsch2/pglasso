try:
    from setuptools import setup
    from setuptools.extension import Extension

except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

from Cython.Build import cythonize
import numpy as np

sourcefiles = ['_pglasso.pyx', '_pglasso.c', 'dpshift.pyx', '__init__.py']


setup(
    name = "pglasso",
    version = "1.0.1",
    author='Maxim Grechkin',
    author_email='grechkin@cs.washington.edu',
    url="https://sites.google.com/a/cs.washington.edu/pathglasso/",
    packages=['pglasso'],
    scripts=["pglasso/__init__.py"],
    description='Pathway Graphical Lasso implementation',
    ext_modules = cythonize( Extension("_pglasso", ["_pglasso.pyx"], include_dirs=[np.get_include()])),
    install_requires=[
        "cython >= 0.20.1",
        "numpy >= 1.8.0",
        "networkx >= 1.8.1"
    ]
)
