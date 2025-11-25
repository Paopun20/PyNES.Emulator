"""
This module defines the Shader decorator for shader programs.
"""

from typing import Optional, Type
from enum import Enum

class ShaderType(Enum):
    """Type of shader program"""

    VERTEX = "vertex"
    FRAGMENT = "fragment"
    GEOMETRY = "geometry"
    COMPUTE = "compute"
    UNKNOWN = "unknown"

class Shader:
    def __init__(
        self,
        description: Optional[str] = "",
        artist: Optional[str] = "Unknown",
        shader_type: Optional[ShaderType] = ShaderType.FRAGMENT,
        validate: bool = True,
        strip_comments: bool = True,
    ) -> None: ...
    def __call__(self, cls: Type) -> type: ...
