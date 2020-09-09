from setuptools import setup, Extension
from Cython.Build import cythonize

import numpy

extentions = [
    Extension(
        "pocketfft",
        sources=["pocketfft/*.pyx"],
        include_dirs=["include/pocketfft", numpy.get_include()],
        extra_compile_args=[],
    )
]

setup(
    name="pocketfft",
    packages=['pocketfft'],
    ext_modules=cythonize(
        extentions, include_path=[numpy.get_include()], annotate=True
    ),
    # needed for cimporting from other  modules
    package_data={'pocketfft': ['*.pxd']},
    zip_safe=False,
    install_requires=[
        'numpy',
        'cython'
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-benchmark',
        ]
    },
)
