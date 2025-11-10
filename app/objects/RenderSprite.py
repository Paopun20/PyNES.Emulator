import moderngl
import numpy as np

DEFAULT_FRAGMENT_SHADER = """
#version 330
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_tex;
void main() {
fragColor = texture(u_tex, v_uv);
}
"""

VERTEX_SHADER = """
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

    def set_fragment_shader(self, shader_src: str):
        """Swap fragment shader at runtime."""
        self.program.release()
        self.fragment_shader = shader_src
        self.program = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=self.fragment_shader)
        self.vao = self.ctx.vertex_array(self.program, [(self.vbo, "2f 2f", "a_pos", "a_uv")], index_buffer=self.ibo)

    def set_uniform(self, name: str, value):
        """Set a uniform in the current shader."""
        if name not in self.program:
            # ignore missing uniforms
            return
        self.program[name].value = value

    def draw(self):
        self.program["u_scale"].value = (self.W, self.H)
        self.texture.use(location=0)
        self.vao.render()

    def destroy(self):
        self.vao.release()
        self.vbo.release()
        self.ibo.release()
        self.texture.release()
        self.program.release()
