from setuptools import (setup, Extension as CythonExtension)
from Cython.Build import cythonize
# from setuptools_rust import RustExtension
from pathlib import Path
from app.__version__ import __version_string__
import numpy as np

# Obtain the absolute path of the directory containing setup.py
ROOT_DIR = Path(__file__).parent.resolve()

cython_ext = cythonize([
    CythonExtension(
        "pynes.mapper",
        sources=["app/pynes/mapper.py"],
        include_dirs=[np.get_include()]
    ),
])

"""
rust_ext = [
    RustExtension(
        target="rust.disklist",
        path="app/rust/disklist/Cargo.toml",
        debug=False,
        py_limited_api=False,
    ),
]
"""

setup(
    name="PyNES",
    version=__version_string__,
    packages=["pynes"],
    ext_modules=cython_ext,
    # rust_extensions=rust_ext,
    # install_requires=list_req,
    include_package_data=True,
    package_dir={'pynes': 'app/pynes'},
    zip_safe=False,
)