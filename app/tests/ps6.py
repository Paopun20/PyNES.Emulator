import moderngl
import numpy as np
from PySide6 import QtGui, QtWidgets, QtOpenGLWidgets


class Canvas(QtOpenGLWidgets.QOpenGLWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ctx = None

    def initializeGL(self):
        self.ctx = moderngl.create_context(required=410)
        
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

        self._program = self.ctx.program(
            vertex_shader='''
                #version 410 core

                in vec2 position;
                in vec3 color;

                out vec4 v_color;

                void main() {
                    gl_Position = vec4(position, 0.0, 1.0);
                    v_color = vec4(color, 1.0);
                }
            ''',
            fragment_shader='''
                #version 410 core

                in vec4 v_color;
                out vec4 fragColor;

                void main() {
                    fragColor = v_color;
                }
            ''',
        )
        
        vbo = self.ctx.buffer(vertices.tobytes())
        cbo = self.ctx.buffer(colors.tobytes())

        self._vao = self.ctx.vertex_array(self._program, [
            (vbo, '2f', 'position'),
            (cbo, '3f', 'color'),
        ])

    def resizeGL(self, width, height):
        super().resizeGL(width, height)

        self.ctx.viewport = (0, 0, width, height)

    def paintGL(self):
        super().paintGL()

        fbo = self.ctx.detect_framebuffer()
        fbo.use()

        self.ctx.clear()

        self._vao.render(moderngl.TRIANGLES)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.resize(120, 100)

        self._canvas = Canvas(self)
        self.setCentralWidget(self._canvas)


fmt = QtGui.QSurfaceFormat()
fmt.setVersion(4, 1)
fmt.setProfile(QtGui.QSurfaceFormat.CoreProfile)
fmt.setDepthBufferSize(24)
QtGui.QSurfaceFormat.setDefaultFormat(fmt)

app = QtWidgets.QApplication()
main_window = MainWindow()
main_window.show()
app.exec()