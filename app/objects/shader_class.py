from __future__ import annotations
import re
import textwrap
from typing import Type, Final
from enum import Enum
from dataclasses import dataclass
from string import Template

__doc_template__ = Template("""
${name}

# Description
${description}

# Uniforms
${uniforms}
""")


class ShaderUniformEnum(Enum):
    FLOAT = "float"
    INT = "int"
    VEC2 = "vec2"
    VEC3 = "vec3"
    VEC4 = "vec4"
    MAT2 = "mat2"
    MAT3 = "mat3"
    MAT4 = "mat4"
    SAMPLER2D = "sampler2D"
    SAMPLER3D = "sampler3D"


@dataclass
class ShaderUniform:
    name: str
    type: ShaderUniformEnum

    def __str__(self):
        return f"uniform {self.type.value} {self.name};"


class Shader:
    """Wraps a Python class docstring as shader code. Usable as a decorator."""
    
    @staticmethod
    def from_file(cls, file_path: str) -> Shader:
        with open(file_path, 'r') as f:
            shader_code = f.read()
        
        class _DynamicShader: __doc__ = shader_code
        return Shader(description=f"Loaded from {file_path}")(_DynamicShader)


    def __init__(self, description: str | None = None):
        self._description: Final[str] = description or ""

    def __call__(self, cls: Type) -> Shader:
        if not cls.__doc__:
            raise ValueError("Shader class must have a docstring containing shader code")

        code = textwrap.dedent(cls.__doc__).strip()
        code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
        code = re.sub(r"//.*", "", code)
        code = re.sub(r"\n\s*\n+", "\n\n", code).strip()

        uniforms = self._parse_uniforms(code)
        uniform_block = "\n".join(str(u) for u in uniforms) or "(none)"

        full_doc = __doc_template__.substitute(
            name=cls.__name__, description=self._description or "(no description)", uniforms=uniform_block
        ).strip()

        self._name: Final[str] = cls.__name__
        self._code: Final[str] = code # .replace("int", "float") # ModernGL have some issues in int
        self._uniforms: Final[list[ShaderUniform]] = uniforms
        self._cls: Final[Type] = cls
        self._doc: Final[str] = full_doc
        self.__doc__ = full_doc

        return self

    def _parse_uniforms(self, code: str) -> list[ShaderUniform]:
        uniforms: list[ShaderUniform] = []
        # Handle both single and array uniforms
        pattern = re.compile(r"\buniform\s+(\w+)\s+(\w+)(?:\[\d+\])?\s*;")
        for match in pattern.finditer(code):
            type_str, name = match.groups()
            try:
                type_enum = ShaderUniformEnum(type_str)
            except ValueError:
                continue
            uniforms.append(ShaderUniform(name, type_enum))
        return uniforms

    @property
    def name(self) -> str:
        return self._name

    @property
    def code(self) -> str:
        return self._code

    @property
    def uniforms(self) -> list[ShaderUniform]:
        return self._uniforms

    def __str__(self) -> str:
        return f"<Shader {self._name} ({len(self._code.splitlines())} lines)>"

    def __repr__(self) -> str:
        return f"<Shader {self._name} ({len(self._code.splitlines())} lines)>"
