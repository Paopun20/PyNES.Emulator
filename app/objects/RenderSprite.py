from __future__ import annotations
import moderngl
import numpy as np
from typing import Final
from shaders.shader_class import Shader

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


class RenderSprite:
    def __init__(self, ctx: moderngl.Context, width=256, height=240, scale=3):
        self.ctx = ctx
        self.width = width
        self.height = height
        self.scale = scale
        self.W = width * scale
        self.H = height * scale
        
        # initial shader program
        self.fragment_shader = DEFAULT_FRAGMENT_SHADER
        self.program = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=self.fragment_shader)

        # create quad
        vertices = np.array(
            [
                0,
                0,
                0,
                0,
                self.W,
                0,
                1,
                0,
                self.W,
                self.H,
                1,
                1,
                0,
                self.H,
                0,
                1,
            ],
            dtype="f4",
        )
        indices = np.array([0, 1, 2, 2, 3, 0], dtype="i4")

        self.vbo = ctx.buffer(vertices.tobytes())
        self.ibo = ctx.buffer(indices.tobytes())
        self.vao = ctx.vertex_array(self.program, [(self.vbo, "2f 2f", "a_pos", "a_uv")], index_buffer=self.ibo)

        # create texture
        self.texture = ctx.texture((width, height), 3, data=np.zeros((height, width, 3), dtype=np.uint8).tobytes())
        self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

    def update_frame(self, frame: np.ndarray):
        """Upload frame (h,w,3 uint8) to GPU."""
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"Frame must be ({self.height},{self.width},3)")
        self.texture.write(frame.tobytes())

    def set_fragment_shader(self, shader_class: Shader):
        """Set a new fragment shader. Falls back to current shader if compilation fails."""
        old_program = self.program
        old_vao = self.vao
        old_shader = self.fragment_shader
        
        try:
            # Clean conversion to string
            new_shader = str(shader_class)
            
            # Compile new program first (before releasing old one)
            new_program = self.ctx.program(
                vertex_shader=VERTEX_SHADER,
                fragment_shader=new_shader
            )
            
            # Create new VAO with new program
            new_vao = self.ctx.vertex_array(
                new_program, [(self.vbo, "2f 2f", "a_pos", "a_uv")], index_buffer=self.ibo
            )
            
            # Success! Now we can safely release old resources
            old_vao.release()
            old_program.release()
            
            # Update to new resources
            self.program = new_program
            self.vao = new_vao
            self.fragment_shader = new_shader
            
        except moderngl.Error as e:
            print(f"Failed to compile fragment shader:\n{str(shader_class)}")
            print(f"Error: {e}")
            print("Keeping previous shader")
            # Don't update anything - old shader remains active
            raise e

    def set_uniform(self, name: str, value):
        """Set a uniform in the current shader."""
        if name not in self.program:
            # ignore missing uniforms
            return
        self.program[name].value = value

    def draw(self):
        """Render the sprite."""
        self.program["u_scale"].value = (self.W, self.H)
        
        # Explicitly bind texture to location 0
        self.texture.use(location=0)
        if "u_tex" in self.program:
            self.program["u_tex"].value = 0
        
        self.vao.render()

    def destroy(self):
        """Release all GPU resources."""
        self.vao.release()
        self.vbo.release()
        self.ibo.release()
        self.texture.release()
        self.program.release()