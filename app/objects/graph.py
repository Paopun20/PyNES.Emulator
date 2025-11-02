import numpy as np
import math
from OpenGL.GL import *
from objects.RenderSprite import RenderSprite

class Graph(RenderSprite):
    def __init__(self, width=400, height=300, x: int = 0, y: int = 0, scale=3):
        super().__init__(width, height, x, y, scale)
        self.clear_color = np.array([255, 255, 255], dtype=np.uint8)
        self.line_color = np.array([255, 0, 0], dtype=np.uint8)
        self._frameBuffer[:] = self.clear_color
        
        # Optional: margins for axis labels/padding
        self.margin_left = 10
        self.margin_right = 10
        self.margin_top = 10
        self.margin_bottom = 10

    def clear(self, color=None):
        """Clear framebuffer."""
        if color is None:
            color = self.clear_color
        self._frameBuffer[:] = color

    def plot_list(self, values: list, color=(255, 0, 0), line_width=1, 
                  auto_scale=True, y_min=None, y_max=None):
        """Plot a list of numeric values as a connected line graph.
        
        Args:
            values: List of numeric values to plot
            color: RGB tuple for line color
            line_width: Thickness of the line (1-5)
            auto_scale: If True, scale to min/max of data
            y_min, y_max: Manual Y-axis range (overrides auto_scale)
        """
        h, w, _ = self._frameBuffer.shape
        n = len(values)
        if n < 2:
            return

        # Calculate plot area (respecting margins)
        plot_x0 = self.margin_left
        plot_x1 = w - self.margin_right
        plot_y0 = self.margin_top
        plot_y1 = h - self.margin_bottom
        plot_width = plot_x1 - plot_x0
        plot_height = plot_y1 - plot_y0

        # Determine Y range
        if y_min is not None and y_max is not None:
            min_v, max_v = y_min, y_max
        elif auto_scale:
            min_v, max_v = min(values), max(values)
        else:
            min_v, max_v = 0, 1
            
        if max_v == min_v:
            max_v = min_v + 1.0  # Avoid division by zero

        # Normalize and map values to pixel coordinates
        ys = []
        xs = []
        for i, v in enumerate(values):
            # Normalize to 0-1 range
            norm = (v - min_v) / (max_v - min_v)
            # Clamp to valid range
            norm = max(0.0, min(1.0, norm))
            # Map to plot area (flip Y since screen Y increases downward)
            y = int(plot_y1 - norm * plot_height)
            x = int(plot_x0 + i * plot_width / (n - 1))
            
            # Ensure coordinates are in bounds
            y = max(0, min(h - 1, y))
            x = max(0, min(w - 1, x))
            
            xs.append(x)
            ys.append(y)

        # Draw line segments
        for i in range(1, n):
            x0, y0 = xs[i-1], ys[i-1]
            x1, y1 = xs[i], ys[i]
            self._draw_line(x0, y0, x1, y1, color, line_width)

    def plot_function(self, func, x_min=-10, x_max=10, samples=200, color=(255, 0, 0)):
        """Plot a mathematical function over a range.
        
        Args:
            func: Callable that takes x and returns y
            x_min, x_max: X-axis range
            samples: Number of sample points
            color: RGB tuple for line color
        """
        xs = np.linspace(x_min, x_max, samples)
        try:
            ys = [func(x) for x in xs]
            # Filter out invalid values (inf, nan)
            valid_points = [(x, y) for x, y in zip(xs, ys) 
                          if math.isfinite(y)]
            if valid_points:
                _, ys_valid = zip(*valid_points)
                self.plot_list(list(ys_valid), color=color)
        except Exception as e:
            print(f"Error plotting function: {e}")

    def draw_axes(self, color=(128, 128, 128)):
        """Draw simple X and Y axes."""
        h, w, _ = self._frameBuffer.shape
        
        # Y-axis (left side)
        x = self.margin_left
        self._draw_line(x, 0, x, h - 1, color)
        
        # X-axis (bottom)
        y = h - self.margin_bottom
        self._draw_line(0, y, w - 1, y, color)

    def draw_grid(self, nx=10, ny=10, color=(220, 220, 220)):
        """Draw a grid with specified number of lines."""
        h, w, _ = self._frameBuffer.shape
        
        plot_x0 = self.margin_left
        plot_x1 = w - self.margin_right
        plot_y0 = self.margin_top
        plot_y1 = h - self.margin_bottom
        
        # Vertical lines
        for i in range(nx + 1):
            x = int(plot_x0 + i * (plot_x1 - plot_x0) / nx)
            self._draw_line(x, plot_y0, x, plot_y1, color)
        
        # Horizontal lines
        for i in range(ny + 1):
            y = int(plot_y0 + i * (plot_y1 - plot_y0) / ny)
            self._draw_line(plot_x0, y, plot_x1, y, color)

    def _draw_line(self, x0, y0, x1, y1, color, width=1):
        """Bresenham line with optional thickness.
        
        Args:
            width: Line thickness (1-5 recommended)
        """
        # Bresenham's line algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            # Draw pixel with thickness
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
        """Draw a pixel with thickness (simple square brush)."""
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
        """Upload framebuffer to GPU texture."""
        self.update_frame(self._frameBuffer)