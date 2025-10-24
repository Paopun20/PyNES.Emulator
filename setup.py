from setuptools import setup, Extension
import numpy as np

try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

ext = '.py' if USE_CYTHON else '.c'

extensions = [
    Extension(
        "pynes.apu",
        ["pynes/apu" + ext],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-ffast-math", "-march=native"],
    ),
    Extension(
        "pynes.cartridge",
        ["pynes/cartridge" + ext],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-ffast-math", "-march=native"],
    ),
    Extension(
        "pynes.controller",
        ["pynes/controller" + ext],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-ffast-math", "-march=native"],
    ),
    Extension(
        "pynes.api.discord",
        ["pynes/api/discord" + ext],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-ffast-math", "-march=native"],
    )
]

if USE_CYTHON:
    extensions = cythonize(extensions)

setup(
    name="PyNES",
    version="0.1.0",
    packages=["pynes", "pynes.helper"],
    ext_modules=extensions,
    install_requires=[
        "numpy",
        "Cython",
        "sounddevice",
        "PyQt5",
    ],
    entry_points={
        "console_scripts": [
            "pynes=pynes.main:main",
        ],
    },
)
