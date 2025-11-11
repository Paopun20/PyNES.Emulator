from __future__ import annotations
import textwrap
from typing import Final, Type, Optional
import re


class Shader:
    """
    Wraps a Python class as a shader object. Can be used as a decorator.

    Example:
        @Shader
        class MyVertexShader:
            '''
            #version 330 core
            void main() { ... }
            '''
    """

    def __init__(self, cls: Type, *, strip_comments: bool = True):
        self._name: Final[str] = cls.__name__
        self._shader_code: str = self._extract_shader_code(
            cls.__doc__, strip_comments
        )
        # Preserve original class for better introspection
        self._original_class: Final[Type] = cls

    def _extract_shader_code(
        self, docstring: Optional[str], strip_comments: bool
    ) -> str:
        if not docstring:
            raise ValueError(f"Shader class must have a docstring containing shader code")

        # Dedent and strip quote marks
        code = textwrap.dedent(docstring).strip()
        code = code.strip('"""').strip("'''").strip()

        if strip_comments:
            # Remove C-style block comments
            code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
            # Remove C-style line comments
            code = re.sub(r"//.*", "", code)

        # Normalize whitespace (collapse multiple blank lines)
        code = re.sub(r"\n\s*\n+", "\n\n", code).strip()

        return code

    @property
    def name(self) -> str:
        """Returns the shader name (class name)."""
        return self._name

    @property
    def code(self) -> str:
        """Returns the cleaned shader code."""
        return self._shader_code

    def __str__(self) -> str:
        return self._shader_code

    def __repr__(self) -> str:
        return f"Shader(name='{self._name}', lines={len(self._shader_code.splitlines())})"