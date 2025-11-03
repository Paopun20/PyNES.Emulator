import numpy as np
import math
from objects.RenderSprite import RenderSprite

class Graph(RenderSprite):
    def __init__(self, ctx, width=400, height=300, scale=3):
        super().__init__(ctx, width, height, scale)
        self._frameBuffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.clear_color = np.array([255, 255, 255], dtype=np.uint8)
        self.line_color = np.array([255, 0, 0], dtype=np.uint8)
        self._frameBuffer[:] = self.clear_color

        self.margin_left = 10
        self.margin_right = 10
        self.margin_top = 10
        self.margin_bottom = 10

    def clear(self, color=None):
        self._frameBuffer[:] = color if color is not None else self.clear_color

    def plot_list(self, values: list, color=(255, 0, 0), line_width=1, auto_scale=True, y_min=None, y_max=None):
        h, w, _ = self._frameBuffer.shape
        n = len(values)
        if n < 2:
            return

        plot_x0 = self.margin_left
        plot_x1 = w - self.margin_right
        plot_y0 = self.margin_top
        plot_y1 = h - self.margin_bottom
        plot_width = plot_x1 - plot_x0
        plot_height = plot_y1 - plot_y0

        if y_min is not None and y_max is not None:
            min_v, max_v = y_min, y_max
        elif auto_scale:
            min_v, max_v = min(values), max(values)
        else:
            min_v, max_v = 0, 1
        if max_v == min_v:
            max_v += 1.0

        xs, ys = [], []
        for i, v in enumerate(values):
            norm = (v - min_v) / (max_v - min_v)
            norm = max(0.0, min(1.0, norm))
            y = int(plot_y1 - norm * plot_height)
            x = int(plot_x0 + i * plot_width / (n - 1))
            y = max(0, min(h - 1, y))
            x = max(0, min(w - 1, x))
            xs.append(x)
            ys.append(y)

        for i in range(1, n):
            self._draw_line(xs[i-1], ys[i-1], xs[i], ys[i], color, line_width)

    def plot_function(self, func, x_min=-10, x_max=10, samples=200, color=(255, 0, 0)):
        xs = np.linspace(x_min, x_max, samples)
        try:
            ys = [func(x) for x in xs]
            valid_points = [(x, y) for x, y in zip(xs, ys) if math.isfinite(y)]
            if valid_points:
                _, ys_valid = zip(*valid_points)
                self.plot_list(list(ys_valid), color=color)
        except Exception as e:
            print(f"Error plotting function: {e}")

    def draw_axes(self, color=(128, 128, 128)):
        h, w, _ = self._frameBuffer.shape
        self._draw_line(self.margin_left, 0, self.margin_left, h - 1, color)
        self._draw_line(0, h - self.margin_bottom, w - 1, h - self.margin_bottom, color)

    def draw_grid(self, nx=10, ny=10, color=(220, 220, 220)):
        h, w, _ = self._frameBuffer.shape
        plot_x0, plot_x1 = self.margin_left, w - self.margin_right
        plot_y0, plot_y1 = self.margin_top, h - self.margin_bottom

        for i in range(nx + 1):
            x = int(plot_x0 + i * (plot_x1 - plot_x0) / nx)
            self._draw_line(x, plot_y0, x, plot_y1, color)
        for i in range(ny + 1):
            y = int(plot_y0 + i * (plot_y1 - plot_y0) / ny)
            self._draw_line(plot_x0, y, plot_x1, y, color)

    def _draw_line(self, x0, y0, x1, y1, color, width=1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            self._draw_thick_pixel(x0, y0, width, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def _draw_thick_pixel(self, x, y, width, color):
        if width == 1:
            if 0 <= x < self.width and 0 <= y < self.height:
                self._frameBuffer[y, x] = color
        else:
            offset = width // 2
            for dy in range(-offset, offset + 1):
                for dx in range(-offset, offset + 1):
                    px, py = x + dx, y + dy
                    if 0 <= px < self.width and 0 <= py < self.height:
                        self._frameBuffer[py, px] = color

    def update(self):
        self.update_frame(self._frameBuffer)