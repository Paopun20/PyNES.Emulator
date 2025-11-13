from __future__ import annotations
import moderngl
import numpy as np
from numpy.typing import NDArray
from typing import Final, TypeAlias
from objects.shader_class import Shader
from logger import log
from enum import Enum

# --- Shader defaults ---
DEFAULT_FRAGMENT_SHADER: Final[str] = """
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
uniform vec2 u_scale;
void main() {
    vec2 ndc = (a_pos / u_scale) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = a_uv;
}
"""

# --- GLSL-style type aliases ---
vec2: TypeAlias = NDArray[np.float32]  # shape (2,)
vec3: TypeAlias = NDArray[np.float32]  # shape (3,)
vec4: TypeAlias = NDArray[np.float32]  # shape (4,)
mat2: TypeAlias = NDArray[np.float32]  # shape (2,2)
mat3: TypeAlias = NDArray[np.float32]  # shape (3,3)
mat4: TypeAlias = NDArray[np.float32]  # shape (4,4)
sampler2D: TypeAlias = int
sampler3D: TypeAlias = int

uniformType: TypeAlias = (
    float
    | int
    | vec2
    | vec3
    | vec4
    | mat2
    | mat3
    | mat4
    | sampler2D
    | sampler3D
)

# --- Renderer class ---
class RenderSprite:
    def __init__(self, ctx: moderngl.Context, width: int = 256, height: int = 240, scale: int = 3):
        self.ctx: moderngl.Context = ctx
        self.width: int = width
        self.height: int = height
        self.scale: int = scale
        self.W: int = width * scale
        self.H: int = height * scale

        self.fragment_shader: str = DEFAULT_FRAGMENT_SHADER
        self.program: moderngl.Program = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=self.fragment_shader)

        # Create quad geometry
        vertices = np.array(
            [
                0, 0, 0, 0,
                self.W, 0, 1, 0,
                self.W, self.H, 1, 1,
                0, self.H, 0, 1,
            ],
            dtype="f4",
        )
        indices = np.array([0, 1, 2, 2, 3, 0], dtype="i4")

        self.vbo = ctx.buffer(vertices.tobytes())
        self.ibo = ctx.buffer(indices.tobytes())
        self.vao = ctx.vertex_array(self.program, [(self.vbo, "2f 2f", "a_pos", "a_uv")], index_buffer=self.ibo)

        # Create blank texture
        self.texture = ctx.texture((width, height), 3, data=np.zeros((height, width, 3), dtype=np.uint8).tobytes())
        self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

    def update_frame(self, frame: NDArray[np.uint8]) -> None:
        """Upload frame (h,w,3 uint8) to GPU."""
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"Frame must be ({self.height},{self.width},3)")
        self.texture.write(frame.tobytes())

    def set_fragment_shader(self, shader_class: Shader) -> None:
        old_program = self.program
        old_vao = self.vao
        old_shader = self.fragment_shader

        try:
            new_shader = str(shader_class.code)
            new_program = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=new_shader)
            new_vao = self.ctx.vertex_array(
                new_program, [(self.vbo, "2f 2f", "a_pos", "a_uv")], index_buffer=self.ibo
            )

            old_vao.release()
            old_program.release()

            self.program = new_program
            self.vao = new_vao
            self.fragment_shader = new_shader

        except moderngl.Error as e:
            log.error(f"Failed to compile fragment shader:\n{str(shader_class.code)}")
            log.error(f"Error:\n{e}")
            log.error(f"Reverting to previous shader.\nPrevious shader code:\n{old_shader}")
            self.program = old_program
            self.vao = old_vao
            self.fragment_shader = old_shader

    def set_uniform(self, name: str, value: uniformType) -> None:
        """Set a uniform in the current shader."""
        if name not in self.program:
            log.warning(f"Uniform {name} not found in shader, ignoring.")
            return
        self.program[name].value = value

    def draw(self) -> None:
        """Render the sprite."""
        self.program["u_scale"].value = (self.W, self.H)
        self.texture.use(location=0)
        if "u_tex" in self.program:
            self.program["u_tex"].value = 0
        self.vao.render()

    def destroy(self) -> None:
        """Release all GPU resources."""
        self.vao.release()
        self.vbo.release()
        self.ibo.release()
        self.texture.release()
        self.program.release()
