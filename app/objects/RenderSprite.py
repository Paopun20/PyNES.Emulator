from typing import Final, List, TypeAlias

import moderngl
import numpy as np
from logger import log
from numpy.typing import NDArray
from objects.shaderclass import Shader, ShaderType, ShaderVariable


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

# GLSL-style type aliases
vec2: TypeAlias = NDArray[np.float32]
vec3: TypeAlias = NDArray[np.float32]
vec4: TypeAlias = NDArray[np.float32]
mat2: TypeAlias = NDArray[np.float32]
mat3: TypeAlias = NDArray[np.float32]
mat4: TypeAlias = NDArray[np.float32]
sampler2D: TypeAlias = int
sampler3D: TypeAlias = int


class RenderSprite:
    def __init__(
        self,
        ctx: moderngl.Context,
        width: int = 256,
        height: int = 240,
        scale: int = 3,
    ):
        self.ctx = ctx
        self.width = width
        self.height = height
        self.scale = scale

        self.W = width * scale
        self.H = height * scale

        self._shclass: Shader = DEFAULT_FRAGMENT_SHADER
        self.fragment_shader: str = DEFAULT_FRAGMENT_SHADER.code

        # Program
        self.program = ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=self.fragment_shader,
        )

        # Geometry (screen quad)
        vertices = np.array(
            [
                0,     0,     0, 0,
                self.W, 0,     1, 0,
                self.W, self.H, 1, 1,
                0,     self.H, 0, 1,
            ],
            dtype="f4",
        )

        indices = np.array([0, 1, 2, 2, 3, 0], dtype="i4")

        self.vbo = ctx.buffer(vertices.tobytes())
        self.ibo = ctx.buffer(indices.tobytes())

        self.vao = ctx.vertex_array(
            self.program,
            [(self.vbo, "2f 2f", "a_pos", "a_uv")],
            index_buffer=self.ibo,
        )

        # RGBA texture
        self.texture = ctx.texture(
            (width, height),
            components=4,
            data=np.zeros((height, width, 4), dtype=np.uint8).tobytes(),
        )
        self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

        # Alpha blending (overlay support)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (
            moderngl.SRC_ALPHA,
            moderngl.ONE_MINUS_SRC_ALPHA,
        )

        # Static uniforms
        if "u_resolution" in self.program:
            self.program["u_resolution"].value = (self.W, self.H)

    def update(self, frame: NDArray[np.uint8]) -> None:
        """Upload RGBA frame (h, w, 4)"""
        if frame.shape != (self.height, self.width, 4):
            raise ValueError(
                f"Frame must be ({self.height}, {self.width}, 4)"
            )
        self.texture.write(frame.tobytes())

    def update_bytes(self, data: bytes) -> None:
        expected = self.width * self.height * 4
        if len(data) != expected:
            raise ValueError(
                f"Invalid RGBA byte size: got {len(data)}, expected {expected} "
                f"({self.width}x{self.height} RGBA)"
            )
        self.texture.write(data)

    def set_fragment_shader(self, shaderclass: Shader) -> bool:
        old_program = self.program
        old_vao = self.vao
        old_shader = self.fragment_shader

        if shaderclass.shader_type != ShaderType.FRAGMENT:
            log.error("Provided shader is not a fragment shader")
            return False

        try:
            new_program = self.ctx.program(
                vertex_shader=VERTEX_SHADER,
                fragment_shader=str(shaderclass.code),
            )

            new_vao = self.ctx.vertex_array(
                new_program,
                [(self.vbo, "2f 2f", "a_pos", "a_uv")],
                index_buffer=self.ibo,
            )

            old_vao.release()
            old_program.release()

            self.program = new_program
            self.vao = new_vao
            self.fragment_shader = shaderclass.code
            self._shclass = shaderclass

            if "u_resolution" in self.program:
                self.program["u_resolution"].value = (self.W, self.H)

            return True

        except moderngl.Error as e:
            log.error("Shader compile failed")
            log.error(e)

            self.program = old_program
            self.vao = old_vao
            self.fragment_shader = old_shader
            return False

    def set_uniform(
        self,
        name: str,
        value: float | int | vec2 | vec3 | vec4 | mat2 | mat3 | mat4,
    ) -> None:
        if name not in self.program:
            return
        self.program[name] = value

    def draw(self, isoverlay: bool = False) -> None:
        if not isoverlay:
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        if "u_tex" in self.program:
            self.program["u_tex"].value = 0

        self.texture.use(location=0)
        self.vao.render()

    def export(self) -> NDArray[np.uint8]:
        data = self.ctx.screen.read(components=4, dtype="f1")
        frame = np.frombuffer(data, dtype=np.uint8).reshape(
            self.H, self.W, 4
        )
        return np.flipud(frame)

    def destroy(self) -> None:
        if self.vao:
            self.vao.release()
        if self.vbo:
            self.vbo.release()
        if self.ibo:
            self.ibo.release()
        if self.texture:
            self.texture.release()
        if self.program:
            self.program.release()
