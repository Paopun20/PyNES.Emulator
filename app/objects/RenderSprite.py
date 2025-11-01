import ctypes
import numpy as np
from OpenGL.GL import *
from typing import Final

VERTEX_SHADER_SRC: Final[str] = """
#version 330 core
in vec2 a_pos;
in vec2 a_uv;
out vec2 v_uv;
uniform vec2 u_scale; // screen scale (width, height)
void main() {
    // a_pos in pixel coords; map to normalized device coords
    vec2 ndc = (a_pos / u_scale) * 2.0 - 1.0;
    // flip Y because ortho/top-left in your main was top-left origin
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = a_uv;
}
"""

DEFAULT_FRAGMENT_SHADER: Final[str] = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_tex;
void main() {
    vec3 c = texture(u_tex, v_uv).rgb;
    fragColor = vec4(c, 1.0);
}
"""

def compile_shader(src: str, shader_type: int) -> int:
    s = glCreateShader(shader_type)
    glShaderSource(s, src)
    glCompileShader(s)
    ok = glGetShaderiv(s, GL_COMPILE_STATUS)
    if not ok:
        err = glGetShaderInfoLog(s).decode(errors="ignore")
        glDeleteShader(s)
        raise RuntimeError(f"Shader compile error:\n{err}\n--- source ---\n{src}")
    return s

def link_program(vs: int, fs: int) -> int:
    p = glCreateProgram()
    glAttachShader(p, vs)
    glAttachShader(p, fs)
    glLinkProgram(p)
    ok = glGetProgramiv(p, GL_LINK_STATUS)
    if not ok:
        err = glGetProgramInfoLog(p).decode(errors="ignore")
        glDeleteProgram(p)
        raise RuntimeError(f"Program link error:\n{err}")
    # detach and delete shaders (they're no longer needed)
    try:
        glDetachShader(p, vs)
        glDetachShader(p, fs)
    except Exception:
        pass
    glDeleteShader(vs)
    glDeleteShader(fs)
    return p

class RenderSprite:
    class UseProgram:
        def __init__(self, program_id):
            self.program_id = program_id
        def __enter__(self):
            glUseProgram(self.program_id)
            return self.program_id
        def __exit__(self, exc_type, exc_val, exc_tb):
            glUseProgram(0)

    def __init__(self, width=256, height=240, scale=1):
        self.width = int(width)
        self.height = int(height)
        self.scale = int(scale)
        self._frameBuffer = np.zeros((height, width, 3), dtype=np.uint8)
        self.current_fragment_shader = DEFAULT_FRAGMENT_SHADER

        # compile & link initial program
        vs = compile_shader(VERTEX_SHADER_SRC, GL_VERTEX_SHADER)
        fs = compile_shader(self.current_fragment_shader, GL_FRAGMENT_SHADER)
        self.program = link_program(vs, fs)

        # attribute/uniform locations (we'll store names but locations can change when re-linking)
        self._update_locations()

        # create VAO/VBO/EBO for a quad in pixel coords (x,y,u,v)
        W = float(self.width * self.scale)
        H = float(self.height * self.scale)
        verts = np.array(
            [
                # x,  y,   u, v
                0.0, 0.0, 0.0, 0.0,
                W,   0.0, 1.0, 0.0,
                W,   H,   1.0, 1.0,
                0.0, H,   0.0, 1.0,
            ],
            dtype=np.float32,
        )
        inds = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

        # VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # VBO
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)

        # EBO
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, inds.nbytes, inds, GL_STATIC_DRAW)

        # set vertex attrib pointers according to current program locations
        self._bind_attrib_pointers()

        glBindVertexArray(0)

        # create texture
        self.tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        # allocate
        empty = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_RGB, GL_UNSIGNED_BYTE, empty)
        glBindTexture(GL_TEXTURE_2D, 0)

    def _update_locations(self):
        # call after (re)linking program to refresh attribute/uniform locations
        self.a_pos = glGetAttribLocation(self.program, "a_pos")
        self.a_uv = glGetAttribLocation(self.program, "a_uv")
        self.u_scale = glGetUniformLocation(self.program, "u_scale")
        self.u_tex = glGetUniformLocation(self.program, "u_tex")
        # keep a cache for other uniforms user may set
        # (we'll query them on demand in set_fragment_config)
    
    def _bind_attrib_pointers(self):
        # bind VAO & buffer then set pointers for whatever locations we have
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        stride = 4 * ctypes.sizeof(ctypes.c_float)  # 4 floats per-vertex
        # a_pos (vec2) at offset 0
        if self.a_pos != -1:
            glEnableVertexAttribArray(self.a_pos)
            glVertexAttribPointer(self.a_pos, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        # a_uv (vec2) at offset 8 bytes
        if self.a_uv != -1:
            glEnableVertexAttribArray(self.a_uv)
            glVertexAttribPointer(self.a_uv, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(2 * ctypes.sizeof(ctypes.c_float)))
        # Note: if program doesn't have those attributes, user won't get correct mapping, but we warned above by -1 check
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def update_frame(self, frame: np.ndarray):
        """Upload frame (h,w,3 uint8). Keep contiguous and correct dtype."""
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        frame_rgb = np.ascontiguousarray(frame)
        if frame_rgb.shape[0] != self.height or frame_rgb.shape[1] != self.width or frame_rgb.shape[2] != 3:
            raise ValueError("Frame shape must be (height, width, 3) matching sprite size")
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        # flip vertically if your data is top-left and you want to keep it consistent with texture coords:
        # frame_rgb = np.flipud(frame_rgb)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb)
        glBindTexture(GL_TEXTURE_2D, 0)

    def set_fragment_shader(self, shader_src: str):
        """Replace the current fragment shader with a new one at runtime.

        Raises RuntimeError if compile/link fails (with shader log included).
        """
        # compile new program with same vertex shader source
        vs = compile_shader(VERTEX_SHADER_SRC, GL_VERTEX_SHADER)
        fs = compile_shader(shader_src, GL_FRAGMENT_SHADER)
        new_program = link_program(vs, fs)

        # if success, swap programs
        old_program = getattr(self, "program", None)
        self.program = new_program
        self.current_fragment_shader = shader_src

        # update attribute/uniform locations and rebind pointers for VAO
        self._update_locations()
        self._bind_attrib_pointers()

        # delete old program (if any)
        if old_program:
            try:
                glDeleteProgram(old_program)
            except Exception:
                pass

    def set_fragment_config(self, shader_config_name: str, value):
        """Set a uniform value in the current fragment shader.

        Supports:
          - float, int
          - (x,y), (x,y,z), (x,y,z,w) lists/tuples/np arrays
          - numpy arrays for matrices (not implemented here but could be extended)
        """
        with self.UseProgram(self.program):
            loc = glGetUniformLocation(self.program, shader_config_name)
            if loc == -1:
                print(f"Warning: uniform '{shader_config_name}' not found in current shader")
                return

            # single float
            if isinstance(value, float):
                glUniform1f(loc, value)
                return
            # single int / sampler index
            if isinstance(value, int) and not isinstance(value, bool):
                glUniform1i(loc, value)
                return

            # list/tuple/numpy array
            if isinstance(value, (list, tuple, np.ndarray)):
                arr = np.asarray(value)
                if arr.dtype.kind in ("U", "S", "O"):
                    raise TypeError("Unsupported array element type for uniform")

                if arr.ndim == 1:
                    if arr.size == 2:
                        glUniform2f(loc, float(arr[0]), float(arr[1]))
                        return
                    elif arr.size == 3:
                        glUniform3f(loc, float(arr[0]), float(arr[1]), float(arr[2]))
                        return
                    elif arr.size == 4:
                        glUniform4f(loc, float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
                        return
                # matrices (2x2,3x3,4x4) -- user can pass flattened array
                if arr.ndim == 2:
                    # handle matrix4x4 for example
                    if arr.shape == (4, 4):
                        glUniformMatrix4fv(loc, 1, GL_FALSE, arr.astype(np.float32).ravel())
                        return
                    if arr.shape == (3, 3):
                        glUniformMatrix3fv(loc, 1, GL_FALSE, arr.astype(np.float32).ravel())
                        return
                    if arr.shape == (2, 2):
                        glUniformMatrix2fv(loc, 1, GL_FALSE, arr.astype(np.float32).ravel())
                        return

            # fallback
            raise TypeError(f"Unsupported uniform type for '{shader_config_name}': {type(value)}")

    def use_program(self):
        return self.UseProgram(self.program)

    def reset_shader(self):
        self.set_fragment_shader(DEFAULT_FRAGMENT_SHADER)

    def draw(self):
        glUseProgram(self.program)
        # set uniforms
        if self.u_scale != -1:
            screen_scale_x = float(self.width * self.scale)
            screen_scale_y = float(self.height * self.scale)
            glUniform2f(self.u_scale, screen_scale_x, screen_scale_y)
        if self.u_tex != -1:
            glUniform1i(self.u_tex, 0)  # texture unit 0

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.tex)

        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)

    def destroy(self):
        try:
            glDeleteTextures([self.tex])
        except Exception:
            pass
        try:
            glDeleteBuffers(1, [self.vbo])
            glDeleteBuffers(1, [self.ebo])
            glDeleteVertexArrays(1, [self.vao])
        except Exception:
            pass
        try:
            glDeleteProgram(self.program)
        except Exception:
            pass
