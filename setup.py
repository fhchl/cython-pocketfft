from setuptools import setup, Extension
from Cython.Build import cythonize

import numpy

extentions = [
    Extension(
        "pocketfft",
        sources=["src/pocketfft.pyx"],
        include_dirs=["include/pocketfft"],
        extra_compile_args=[],
    )
]

setup(
    name="pocketfft",
    ext_modules=cythonize(
        extentions, include_path=[numpy.get_include()], annotate=True
    ),
)
