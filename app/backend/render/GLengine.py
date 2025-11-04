import moderngl
import numpy as np
from PySide6 import QtGui, QtWidgets, QtOpenGLWidgets


class Shader:
    def __init__(self, ctx, vertex_source, fragment_source):
        """Create a shader program from vertex and fragment shader sources."""
        self.ctx = ctx
        self.program = ctx.program(
            vertex_shader=vertex_source,
            fragment_shader=fragment_source
        )
    
    def __getitem__(self, name):
        """Access uniforms by name."""
        return self.program[name]
    
    def release(self):
        """Release shader resources."""
        if self.program:
            self.program.release()


class GLEngine:
    def __init__(self, ctx):
        """Initialize the GL engine with a ModernGL context."""
        self.ctx = ctx
        self.shader = None
        self.vao = None
        self._vertex_buffer = None
        self._color_buffer = None
        
    def setup(self):
        """Setup shaders, buffers, and VAO."""
        # Create shader program
        vertex_shader = '''
            #version 410 core

            in vec2 position;
            in vec3 color;

            out vec4 v_color;

            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
                v_color = vec4(color, 1.0);
            }
        '''
        
        fragment_shader = '''
            #version 410 core

            in vec4 v_color;
            out vec4 fragColor;

            void main() {
                fragColor = v_color;
            }
        '''
        
        self.shader = Shader(self.ctx, vertex_shader, fragment_shader)
        
        # Create vertex data
        vertices = np.array([
            (-0.6, -0.6),
            (0.6, -0.6),
            (0.0, 0.6),
        ], dtype='f4')
        
        colors = np.array([
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ], dtype='f4')
        
        # Create buffers
        self._vertex_buffer = self.ctx.buffer(vertices.tobytes())
        self._color_buffer = self.ctx.buffer(colors.tobytes())
        
        # Create VAO
        self.vao = self.ctx.vertex_array(self.shader.program, [
            (self._vertex_buffer, '2f', 'position'),
            (self._color_buffer, '3f', 'color'),
        ])
    
    def resize(self, width, height):
        """Handle viewport resize."""
        self.ctx.viewport = (0, 0, width, height)
    
    def render(self):
        """Render the scene."""
        fbo = self.ctx.detect_framebuffer()
        fbo.use()
        
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        
        if self.vao:
            self.vao.render(moderngl.TRIANGLES)
    
    def release(self):
        """Release all GL resources."""
        if self.vao:
            self.vao.release()
        if self._vertex_buffer:
            self._vertex_buffer.release()
        if self._color_buffer:
            self._color_buffer.release()
        if self.shader:
            self.shader.release()


class GLWidget(QtOpenGLWidgets.QOpenGLWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = None
        self.ctx = None
    
    def initializeGL(self):
        """Initialize OpenGL context and engine."""
        self.ctx = moderngl.create_context(require=410)
        self.engine = GLEngine(self.ctx)
        self.engine.setup()
    
    def resizeGL(self, width, height):
        """Handle widget resize."""
        super().resizeGL(width, height)
        if self.engine:
            self.engine.resize(width, height)
    
    def paintGL(self):
        """Paint the GL content."""
        super().paintGL()
        if self.engine:
            self.engine.render()
    
    def cleanup(self):
        """Clean up resources."""
        if self.engine:
            self.engine.release()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.resize(800, 600)
        self.setWindowTitle("ModernGL Triangle")
        
        self._gl_widget = GLWidget(self)
        self.setCentralWidget(self._gl_widget)
    
    def closeEvent(self, event):
        """Handle window close event."""
        self._gl_widget.cleanup()
        super().closeEvent(event)