"""This is an fix for type checkers."""

from typing import Optional, Type

class ShaderType: ...

class Shader:
    def __init__(
        self,
        description: Optional[str] = "",
        artist: Optional[str] = "Unknown",
        shader_type: Optional[ShaderType] = None,
        validate: bool = True,
        strip_comments: bool = True,
    ) -> None: ...
    def __call__(self, cls: Type) -> type: ...