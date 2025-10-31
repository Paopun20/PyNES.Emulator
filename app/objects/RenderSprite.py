import ctypes

import numpy as np
from OpenGL.GL import *

VERTEX_SHADER_SRC = """
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

FRAGMENT_SHADER_SRC = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_tex;
void main() {
    vec3 c = texture(u_tex, v_uv).rgb;
    fragColor = vec4(c, 1.0);
}
"""


def compile_shader(src, shader_type):
    s = glCreateShader(shader_type)
    glShaderSource(s, src)
    glCompileShader(s)
    ok = glGetShaderiv(s, GL_COMPILE_STATUS)
    if not ok:
        err = glGetShaderInfoLog(s).decode()
        raise RuntimeError(f"Shader compile error: {err}")
    return s


def link_program(vs, fs):
    p = glCreateProgram()
    glAttachShader(p, vs)
    glAttachShader(p, fs)
    glLinkProgram(p)
    ok = glGetProgramiv(p, GL_LINK_STATUS)
    if not ok:
        err = glGetProgramInfoLog(p).decode()
        raise RuntimeError(f"Program link error: {err}")
    # detach/delete shaders
    glDetachShader(p, vs)
    glDetachShader(p, fs)
    glDeleteShader(vs)
    glDeleteShader(fs)
    return p


class RenderSprite:
    def __init__(self, width=256, height=240, scale=1):
        self.width = width
        self.height = height
        self.scale = scale
        self._frameBuffer = np.zeros((height, width, 3), dtype=np.uint8)
        self.current_fragment_shader = FRAGMENT_SHADER_SRC

        # compile shader program
        vs = compile_shader(VERTEX_SHADER_SRC, GL_VERTEX_SHADER)
        fs = compile_shader(FRAGMENT_SHADER_SRC, GL_FRAGMENT_SHADER)
        self.program = link_program(vs, fs)

        # attribute/uniform locations
        self.a_pos = glGetAttribLocation(self.program, "a_pos")
        self.a_uv = glGetAttribLocation(self.program, "a_uv")
        self.u_scale = glGetUniformLocation(self.program, "u_scale")
        self.u_tex = glGetUniformLocation(self.program, "u_tex")

        # VBO for a single quad in pixel coords (x,y, u,v)
        # We'll create vertices for a quad covering [0..width*scale] x [0..height*scale]
        W = float(width * scale)
        H = float(height * scale)
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

        # create VAO, VBO, EBO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)

        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, inds.nbytes, inds, GL_STATIC_DRAW)

        # position (vec2)
        stride = 4 * ctypes.sizeof(ctypes.c_float)
        glEnableVertexAttribArray(self.a_pos)
        glVertexAttribPointer(
            self.a_pos,
            2,
            GL_FLOAT,
            GL_FALSE,
            stride,
            ctypes.c_void_p(0),
        )
        # uv (vec2)
        glEnableVertexAttribArray(self.a_uv)
        glVertexAttribPointer(
            self.a_uv,
            2,
            GL_FLOAT,
            GL_FALSE,
            stride,
            ctypes.c_void_p(2 * ctypes.sizeof(ctypes.c_float)),
        )

        glBindVertexArray(0)

        # create texture once
        self.tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        # important for tight rows
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        # allocate texture (no data yet or empty_data)
        empty = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB,
            self.width,
            self.height,
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            empty,
        )
        glBindTexture(GL_TEXTURE_2D, 0)

    def update_frame(self, frame):
        """Upload frame (numpy array h,w,3 uint8). Keep it contiguous and correct dtype."""
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        frame_rgb = np.ascontiguousarray(frame)  # ensure contiguity
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        # If your source is top-left and you want it to appear correctly, you may need to flip vertically:
        # frame_rgb = np.flipud(frame_rgb)
        glTexSubImage2D(
            GL_TEXTURE_2D,
            0,
            0,
            0,
            self.width,
            self.height,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            frame_rgb,
        )
        glBindTexture(GL_TEXTURE_2D, 0)
    
    def set_fragment_shader(self, shader_src: str):
        """Replace the current fragment shader with a new one."""
        # Store the new shader source
        self.current_fragment_shader = shader_src
        
        # Recreate the program
        vs = compile_shader(VERTEX_SHADER_SRC, GL_VERTEX_SHADER)
        fs = compile_shader(shader_src, GL_FRAGMENT_SHADER)
        
        # Delete old program
        old_program = self.program
        
        # Create new program
        self.program = link_program(vs, fs)
        
        # Delete old program
        glDeleteProgram(old_program)
        
        # Update uniform/attribute locations
        self.a_pos = glGetAttribLocation(self.program, "a_pos")
        self.a_uv = glGetAttribLocation(self.program, "a_uv")
        self.u_scale = glGetUniformLocation(self.program, "u_scale")
        self.u_tex = glGetUniformLocation(self.program, "u_tex")
    
    def reset_shader(self):
        """Reset to the default fragment shader."""
        self.set_fragment_shader(FRAGMENT_SHADER_SRC)

    def draw(self):
        glUseProgram(self.program)
        # set uniforms
        screen_scale = np.array(
            [self.width * self.scale, self.height * self.scale], dtype=np.float32
        )
        glUniform2f(self.u_scale, float(screen_scale[0]), float(screen_scale[1]))
        glUniform1i(self.u_tex, 0)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.tex)

        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)

    def destroy(self):
        glDeleteTextures([self.tex])
        glDeleteBuffers(1, [self.vbo])
        glDeleteBuffers(1, [self.ebo])
        glDeleteVertexArrays(1, [self.vao])
        glDeleteProgram(self.program)