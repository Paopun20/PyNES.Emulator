import sys
from pathlib import Path

import numpy as np
import pybind11
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools import Extension as CythonExtension

from app.__version__ import __version_string__

ROOT_DIR = Path(__file__).parent.resolve() / "app"

cython_ext = cythonize(
    [
        CythonExtension("pynes.mapper", sources=[ROOT_DIR / "pynes" / "mapper.py"], include_dirs=[np.get_include()]),
    ]
)

extra_args = []
if sys.platform.startswith("win"):
    extra_args.append("/std:c++17")
else:
    extra_args.append("-std=c++17")

"""
clipboard_ext = [
    Extension(
        "app.util.Clipboard",
        sources=[ROOT_DIR / "extension" / "cpp" / "Clipboard" / "Clipboard.cpp"],
        include_dirs=[
            pybind11.get_include(),
            np.get_include(),
            str(ROOT_DIR)
        ],
        language="c++",
        libraries=["User32"],
        extra_compile_args=extra_args,
    )
]
"""

setup(
    name="PyNES",
    version=__version_string__,
    packages=["pynes"],
    ext_modules=cython_ext,  # + clipboard_ext,
    include_package_data=True,
    package_dir={"pynes": "app/pynes"},
    zip_safe=False,
)
