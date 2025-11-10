from PySide6.QtGui import QImage, 
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QOpenGLTexture import QOpenGLTexture
import numpy as np
from OpenGL.GL import *

class Canvas(QOpenGLWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.width, self.height = 640, 480
        self.setFixedSize(self.width, self.height)

        # create a blank image (RGB32)
        self.image = QImage(self.width, self.height, QImage.Format_RGB32)
        self.image.fill(0x00000000)  # black
        self.texture = None

    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_TEXTURE_2D)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)

        # Convert QImage to texture if not done
        if self.texture is None:
            self.texture = QOpenGLTexture(self.image)

        # Bind texture
        self.texture.bind()

        # Simple quad covering the viewport
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(-1, -1)
        glTexCoord2f(1, 0); glVertex2f(1, -1)
        glTexCoord2f(1, 1); glVertex2f(1, 1)
        glTexCoord2f(0, 1); glVertex2f(-1, 1)
        glEnd()
        self.texture.release()

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        self.width, self.height = width, height
    
    def update_frame(self, frame: np.ndarray):
        """Upload frame (h,w,3 uint8) to GPU."""
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"Frame must be ({self.height},{self.width},3)")

        # Convert to QImage without copying if possible
        self.image = QImage(frame.data, self.width, self.height, 3 * self.width, QImage.Format_RGB888)

        # If texture already exists, update it
        if self.texture is not None:
            self.texture.destroy()  # delete old texture
            self.texture = QOpenGLTexture(self.image)
        else:
            self.texture = QOpenGLTexture(self.image)

        self.update()  # schedule paintGL


# Demo application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QMainWindow()
    canvas = Canvas()
    window.setCentralWidget(canvas)
    window.show()

    import time

    # Simple animation loop
    def animate():
        while True:
            frame = np.random.randint(0, 255, (canvas.height, canvas.width, 3), dtype=np.uint8)
            canvas.update_frame(frame)
            app.processEvents()  # allow GUI to update
            time.sleep(1/30)    # 30 FPS

    import threading
    threading.Thread(target=animate, daemon=True).start()

    sys.exit(app.exec())