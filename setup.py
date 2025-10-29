from setuptools import (setup, Extension as CythonExtension)
from Cython.Build import cythonize
from setuptools_rust import RustExtension
import numpy as np

cython_ext = cythonize([
    CythonExtension(
        name="pynes.api.discord",
        sources=["app/pynes/api/discord.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-ffast-math", "-march=native"],
    )
])

rust_ext = [
    RustExtension(
        target="disklist",
        path="app/pynes/rust/disklist/Cargo.toml",
        debug=False,
        py_limited_api=False,
    ),
]

list_req = []
with open("requirements.txt") as f:
    list_req = [line.strip() for line in f]

setup(
    name="PyNES",
    version="0.0.0",
    packages=["pynes"],
    ext_modules=cython_ext,
    rust_extensions=rust_ext,
    install_requires=list_req,
    # include_package_data=True,
    package_dir={'pynes': 'app/pynes'},
    zip_safe=False,
)
