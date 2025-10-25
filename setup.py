from setuptools import (setup, Extension as CythonExtension)
from Cython.Build import cythonize
from setuptools_rust import RustExtension
import numpy as np

cython_ext = cythonize([
    CythonExtension(
        name="pynes.api.discord",
        sources=["pynes/api/discord.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-ffast-math", "-march=native"],
    )
])

rust_ext = [
    RustExtension(
        target="pynes.rust.disklist",
        path="pynes/rust/disklist/Cargo.toml",
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
    entry_points={
        "console_scripts": ["pynes=pynes.main:main"],
    },
    zip_safe=False,
)
