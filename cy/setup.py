from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Horizontal Adjacency',
    ext_modules=cythonize("cy_horizontal_adjacency.pyx"),
    zip_safe=False,
)