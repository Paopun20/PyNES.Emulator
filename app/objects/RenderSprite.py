from typing import Final, TypeAlias
from weakref import WeakValueDictionary

import moderngl
import numpy as np
from logger import log
from numpy.typing import NDArray
from objects.shaderclass import Shader, ShaderType


# Shader defaults
@Shader("Default fragment shader for rendering", "Build-in Shader", ShaderType.FRAGMENT)
class DEFAULT_FRAGMENT_SHADER:
    """
    #version 330
    in vec2 v_uv;
    out vec4 fragColor;
    uniform sampler2D u_tex;
    void main() {
        fragColor = texture(u_tex, v_uv);
    }
    """


VERTEX_SHADER: Final[str] = """
#version 330
in vec2 a_pos;
in vec2 a_uv;
out vec2 v_uv;

uniform vec2 u_resolution;

void main() {
    vec2 ndc = (a_pos / u_resolution) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = a_uv;
}
"""

# Module-level constants
_QUAD_INDICES: Final[np.ndarray] = np.array([0, 1, 2, 2, 3, 0], dtype="i4")
_QUAD_INDICES_BYTES: Final[bytes] = _QUAD_INDICES.tobytes()

# GLSL-style type aliases
vec2: TypeAlias = NDArray[np.float32]
vec3: TypeAlias = NDArray[np.float32]
vec4: TypeAlias = NDArray[np.float32]
mat2: TypeAlias = NDArray[np.float32]
mat3: TypeAlias = NDArray[np.float32]
mat4: TypeAlias = NDArray[np.float32]
sampler2D: TypeAlias = int
sampler3D: TypeAlias = int

# Shader cache using weak references
_shader_cache: WeakValueDictionary[int, moderngl.Program] = WeakValueDictionary()


class RenderSprite:
    __slots__ = (
        "ctx",
        "width",
        "height",
        "scale",
        "W",
        "H",
        "_shclass",
        "fragment_shader",
        "_expected_shape",
        "_expected_bytes",
        "fast_mode",
        "program",
        "vbo",
        "ibo",
        "vao",
        "texture",
        "_resolution_tuple",
    )

    def __init__(
        self,
        ctx: moderngl.Context,
        width: int = 256,
        height: int = 240,
        scale: int = 3,
        fast_mode: bool = False,
    ):
        self.ctx = ctx
        self.width = width
        self.height = height
        self.scale = scale
        self.fast_mode = fast_mode

        self.W = width * scale
        self.H = height * scale
        self._resolution_tuple = (self.W, self.H)

        self._shclass: Shader = DEFAULT_FRAGMENT_SHADER
        self.fragment_shader: str = DEFAULT_FRAGMENT_SHADER.code

        # Cache for validation
        self._expected_shape = (height, width, 4)
        self._expected_bytes = width * height * 4

        # Initialize OpenGL resources
        self._init_program()
        self._init_geometry()
        self._init_texture()
        self._setup_blending()

    def _init_program(self) -> None:
        """Initialize shader program"""
        shader_hash = hash(self.fragment_shader)

        # Check cache first
        if shader_hash in _shader_cache:
            self.program = _shader_cache[shader_hash]
        else:
            self.program = self.ctx.program(
                vertex_shader=VERTEX_SHADER,
                fragment_shader=self.fragment_shader,
            )
            _shader_cache[shader_hash] = self.program

        # Set static uniforms once
        if "u_resolution" in self.program:
            self.program["u_resolution"] = self._resolution_tuple
        if "u_tex" in self.program:
            self.program["u_tex"] = 0

    def _init_geometry(self) -> None:
        """Initialize vertex buffers and VAO"""
        # Create vertices once as bytes
        vertices = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                float(self.W),
                0.0,
                1.0,
                0.0,
                float(self.W),
                float(self.H),
                1.0,
                1.0,
                0.0,
                float(self.H),
                0.0,
                1.0,
            ],
            dtype="f4",
        )

        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.ibo = self.ctx.buffer(_QUAD_INDICES_BYTES)

        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.vbo, "2f 2f", "a_pos", "a_uv")],
            index_buffer=self.ibo,
        )

    def _init_texture(self) -> None:
        """Initialize texture with optimal settings"""
        self.texture = self.ctx.texture(
            (self.width, self.height),
            components=4,
        )
        self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

    def _setup_blending(self) -> None:
        """Configure alpha blending"""
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (
            moderngl.SRC_ALPHA,
            moderngl.ONE_MINUS_SRC_ALPHA,
        )

    def update(self, frame: NDArray[np.uint8]) -> None:
        """Upload RGBA frame (h, w, 4)"""
        if not self.fast_mode:
            if frame.shape != self._expected_shape:
                raise ValueError(f"Frame must be {self._expected_shape}, got {frame.shape}")
        self.texture.write(frame.tobytes())

    def update_bytes(self, data: bytes) -> None:
        """Upload raw RGBA bytes directly"""
        if not self.fast_mode:
            data_len = len(data)
            if data_len != self._expected_bytes:
                raise ValueError(f"Invalid RGBA byte size: got {data_len}, expected {self._expected_bytes}")
        self.texture.write(data)

    def set_fragment_shader(self, shaderclass: Shader) -> bool:
        """Dynamically replace fragment shader"""
        if shaderclass.shader_type != ShaderType.FRAGMENT:
            log.error("Provided shader is not a fragment shader")
            return False

        shader_hash = hash(shaderclass.code)

        # Check cache
        if shader_hash in _shader_cache:
            new_program = _shader_cache[shader_hash]
        else:
            try:
                new_program = self.ctx.program(
                    vertex_shader=VERTEX_SHADER,
                    fragment_shader=str(shaderclass.code),
                )
                _shader_cache[shader_hash] = new_program
            except moderngl.Error as e:
                log.error(f"Shader compile failed: {e}")
                return False

        # Create new VAO
        try:
            new_vao = self.ctx.vertex_array(
                new_program,
                [(self.vbo, "2f 2f", "a_pos", "a_uv")],
                index_buffer=self.ibo,
            )
        except Exception as e:
            log.error(f"VAO creation failed: {e}")
            return False

        # Clean up old resources
        old_vao = self.vao
        old_program = self.program

        # Update state
        self.program = new_program
        self.vao = new_vao
        self.fragment_shader = shaderclass.code
        self._shclass = shaderclass

        # Release old VAO (program stays in cache)
        old_vao.release()

        # Only release old program if not in cache
        if hash(self.fragment_shader) not in _shader_cache:
            old_program.release()

        # Set uniforms for new program
        if "u_resolution" in self.program:
            self.program["u_resolution"] = self._resolution_tuple
        if "u_tex" in self.program:
            self.program["u_tex"] = 0

        return True

    def set_uniform(
        self,
        name: str,
        value: float | int | vec2 | vec3 | vec4 | mat2 | mat3 | mat4,
    ) -> None:
        """Set a uniform value for custom shaders"""
        if name in self.program:
            self.program[name].value = value  # type: ignore

    def reset_fragment_shader(self) -> bool:
        """Reset to default fragment shader"""
        return self.set_fragment_shader(DEFAULT_FRAGMENT_SHADER)

    def draw(self, isoverlay: bool = False) -> None:
        """Render the sprite"""
        if not isoverlay:
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        self.texture.use(location=0)
        self.vao.render()

    def export(self) -> NDArray[np.uint8]:
        """Export current framebuffer as RGBA array"""
        data = self.ctx.screen.read(components=4, dtype="f1")
        frame = np.frombuffer(data, dtype=np.uint8).reshape(self.H, self.W, 4)
        return np.flipud(frame)

    def destroy(self) -> None:
        """Clean up OpenGL resources"""
        if hasattr(self, "vao") and self.vao:
            self.vao.release()
        if hasattr(self, "vbo") and self.vbo:
            self.vbo.release()
        if hasattr(self, "ibo") and self.ibo:
            self.ibo.release()
        if hasattr(self, "texture") and self.texture:
            self.texture.release()
