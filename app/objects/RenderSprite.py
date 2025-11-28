from xmlrpc.client import Boolean
import moderngl
import numpy as np
from numpy.typing import NDArray
from typing import Any, Final, TypeAlias, List, Tuple
from objects.shaderclass import Shader, ShaderVariable, ShaderType
from logger import log

# --- Shader defaults ---
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
    ndc.y = -ndc.y; // Flip Y axis for ModernGL
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


# --- Renderer class ---
class RenderSprite:
    def __init__(self, ctx: moderngl.Context, width: int = 256, height: int = 240, scale: int = 3):
        self.ctx: moderngl.Context = ctx
        self.width: int = width
        self.height: int = height
        self.scale: int = scale
        self.W: int = width * scale
        self.H: int = height * scale
        self._shclass: Shader = DEFAULT_FRAGMENT_SHADER

        self.fragment_shader: str = DEFAULT_FRAGMENT_SHADER.code
        self.program: moderngl.Program = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=self.fragment_shader)

        # Create quad geometry
        vertices: NDArray[np.float32] = np.array(
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
        indices: NDArray[np.int32] = np.array([0, 1, 2, 2, 3, 0], dtype="i4")

        self.vbo: moderngl.Buffer = ctx.buffer(vertices.tobytes())
        self.ibo: moderngl.Buffer = ctx.buffer(indices.tobytes())
        self.vao: moderngl.VertexArray = ctx.vertex_array(
            self.program, [(self.vbo, "2f 2f", "a_pos", "a_uv")], index_buffer=self.ibo
        )

        # Create blank texture
        self.texture: moderngl.Texture = ctx.texture(
            (width, height), 3, data=np.zeros((height, width, 3), dtype=np.uint8).tobytes()
        )
        self.texture.filter: Tuple[int, int] = (moderngl.NEAREST, moderngl.NEAREST)

    def update_frame(self, frame: NDArray[np.uint8]) -> None:
        """Upload frame (h,w,3 uint8) to GPU."""
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"Frame must be ({self.height},{self.width},3)")
        self.texture.write(frame.tobytes())

    def set_fragment_shader(self, shadercass: Shader) -> bool:
        old_program = self.program
        old_vao = self.vao
        old_shader = self.fragment_shader

        if shadercass.shader_type != ShaderType.FRAGMENT:
            log.error("Provided shader is not a fragment shader. Aborting shader change.")
            return False

        try:
            new_shader = str(shadercass.code)
            new_program = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=new_shader)
            new_vao = self.ctx.vertex_array(new_program, [(self.vbo, "2f 2f", "a_pos", "a_uv")], index_buffer=self.ibo)

            old_vao.release()
            old_program.release()

            self.program = new_program
            self.vao = new_vao
            self.fragment_shader = new_shader
            self._shclass = shadercass
            return True
        except moderngl.Error as e:
            log.error(f"Failed to compile fragment shader:\n{str(shadercass.code)}")
            log.error(f"Error:\n{e}")
            log.error("Reverting to previous shader.\nPrevious shader.")
            self.program = old_program
            self.vao = old_vao
            self.fragment_shader = old_shader
            return False

    def get_current_shader_name(self) -> str:
        """Get the name of the current shader."""
        return self._shclass.name

    def get_current_shader_config(self) -> List[ShaderVariable]:
        """Get the configuration of the current shader."""
        return self._shclass.uniforms

    def reset_fragment_shader(self) -> None:
        """Reset to the default fragment shader."""
        self.set_fragment_shader(DEFAULT_FRAGMENT_SHADER)

    def set_uniform(
        self, name: str, value: float | int | vec2 | vec3 | vec4 | mat2 | mat3 | mat4 | sampler2D | sampler3D
    ) -> None:
        """Set a uniform in the current shader."""
        if name not in self.program:
            # log.warning(f"Uniform {name} not found in shader, ignoring.")
            return

        # Fixed: Check for actual types instead of type aliases
        if isinstance(value, (float, int, np.ndarray)):
            self.program[name] = value
        else:
            log.error(f"Invalid uniform type: {type(value)}")

    def draw(self) -> None:
        """Render the sprite."""
        if self.texture is None:
            log.error("Texture is None")
            return
        if self.program is None:
            log.error("Program is None")
            return
        if self.vao is None:
            log.error("VAO is None")
            return

        if "u_resolution" in self.program:
            self.program["u_resolution"].value = (self.W, self.H)

        if "u_tex" in self.program:
            self.program["u_tex"].value = 0

        self.texture.use(location=0)
        self.vao.render()

    def export(self) -> NDArray[np.uint8]:
        """Export the current frame buffer as a NumPy array."""
        # The rendered output is in ctx.screen, not in the texture attachment
        data = self.ctx.screen.read(components=3, dtype="f1")

        # Convert bytes to NumPy array and reshape
        frame = np.frombuffer(data, dtype=np.uint8).reshape(self.H, self.W, 3)

        # Flip vertically because OpenGL coordinates are bottom-up
        frame = np.flipud(frame)

        return frame

    def destroy(self) -> None:
        """Release all GPU resources."""
        if hasattr(self, 'vao') and self.vao:
            self.vao.release()
        if hasattr(self, 'vbo') and self.vbo:
            self.vbo.release()
        if hasattr(self, 'ibo') and self.ibo:
            self.ibo.release()
        if hasattr(self, 'texture') and self.texture:
            self.texture.release()
        if hasattr(self, 'program') and self.program:
            self.program.release()
