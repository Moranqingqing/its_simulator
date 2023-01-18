import numpy as np
import cairocffi as cairo
from gym.envs.classic_control import rendering

w = 300
h = 300

surface = cairo.ImageSurface (cairo.FORMAT_ARGB32, w, h)
ctx = cairo.Context (surface)

# Draw out the triangle using absolute coordinates
ctx.move_to (w/2, h/3)
ctx.line_to (2*w/3, 2*h/3)
ctx.rel_line_to (-1*w/3, 0)
ctx.close_path()
ctx.set_source_rgb (0, 0, 0)  # black
ctx.set_line_width(15)
ctx.stroke()

buf = surface.get_data()
array = np.ndarray (shape=(w,h,4), dtype=np.uint8, buffer=buf)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
imgplot = plt.imshow(array)
plt.show()

viewer = rendering.SimpleImageViewer()
viewer.imshow(array)