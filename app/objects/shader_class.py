from __future__ import annotations
import re, textwrap
from typing import Type, Final, Optional


class Shader:
    """Wraps a Python class docstring as shader code. Usable as a decorator."""

    def __init__(self, cls: Type, *, strip_comments: bool = True):
        if not cls.__doc__:
            raise ValueError("Shader class must have a docstring containing shader code")

        code = textwrap.dedent(cls.__doc__).strip()
        if strip_comments:
            code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
            code = re.sub(r"//.*", "", code)
        code = re.sub(r"\n\s*\n+", "\n\n", code).strip()

        self._name: Final[str] = cls.__name__
        self._code: Final[str] = code
        self._cls: Final[Type] = cls  # keep reference for introspection

    @property
    def name(self) -> str:
        return self._name

    @property
    def code(self) -> str:
        return self._code

    def __str__(self) -> str:
        return self._code

    def __repr__(self) -> str:
        return f"<Shader {self._name} ({len(self._code.splitlines())} lines)>"
