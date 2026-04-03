"""
2D rendering of Dynamic Multi-dimensional Capability Resource Scheduling (DMCRS) domain
"""

import math
import os
import sys

from enum import Enum
import numpy as np
import six
from gymnasium import error

if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"
        # (JDS 2016/04/15): special environment variable settings for Apple systems to avoid bug on Anaconda 2.3.0 / Yosemite


try:
    import pyglet
except ImportError:
    raise ImportError(
        """
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    """
    )

try:
    from pyglet.gl import *
except ImportError:
    raise ImportError(
        """
    Error occured while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    """
    )


RAD2DEG = 57.29577951308232
# # Define some colors
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)
_RED = (255, 0, 0)
_BLUE = (0, 0, 255)
_YELLOW = (255, 255, 0)
_ORANGE = (255, 165, 0)
_PURPLE = (128, 0, 128)
_LIGHT_BLUE = (173, 216, 230)

_BACKGROUND_COLOR = _WHITE
_GRID_COLOR = _BLACK

from .environment import TaskDynamicType

def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            "Invalid display specification: {}. (Must be a string like :0 or None.)".format(
                spec
            )
        )


class Viewer(object): # Visualization Renderer Class - responsible for creating windows and rendering all elements in the environment
    def __init__(self, world_size): # world_size: The size of the world, in the format of (number of rows, number of columns)
        display = get_display(None)
        self.rows, self.cols = world_size
        
        # set the size of grid and icon
        self.grid_size = 60
        self.icon_size = 20
        # set the size of window
        self.width = 1 + self.cols * (self.grid_size + 1)
        self.height = 1 + self.rows * (self.grid_size + 1)
        self.window = pyglet.window.Window(
            width=self.width, height=self.height, display=display
        )
        self.window.on_close = self.window_closed_by_user
        self.isopen = True
        # Enable OpenGL blending function for translucent effects
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # Load image resources
        script_dir = os.path.dirname(__file__)
        pyglet.resource.path = [os.path.join(script_dir, "icons")] # set path of resourses
        pyglet.resource.reindex()

        self.img_apple = pyglet.resource.image("apple.png")
        self.img_agent = pyglet.resource.image("agent.png")

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.isopen = False
        exit()

    def set_bounds(self, left, right, bottom, top): # Set rendering boundary
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(
            translation=(-left * scalex, -bottom * scaley), scale=(scalex, scaley)
        )

    def render(self, env, return_rgb_array=False):
        """
        The main methods of rendering the environment

        Parameters:
            env: The environment object to be rendered
            return_rgb_array: Whether to return the RGB array instead of displaying the graph
        return:
            If `return_rgb_array` is set to `True`, the RGB array representation of the environment will be returned;
            Otherwise, return the status of whether the window is open (Boolean value)
        """
        glClearColor(*_WHITE, 0)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self._draw_grid()
        self._draw_resource_sight_overlay(env)
        self._draw_tasks(env)
        self._draw_resources(env)

        time_step_label = pyglet.text.Label(
            f"Step: {env.current_step}",
            font_name="Times New Roman",
            font_size=14,
            bold=True,
            x=20,
            y=self.height - 20,
            anchor_x="left",
            anchor_y="top",
            color=(*_BLUE, 255),
        )
        time_step_label.draw()

        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        self.window.flip()
        return arr if return_rgb_array else self.isopen

    def _draw_grid(self):
        batch = pyglet.graphics.Batch()
        # vertical lines
        for r in range(self.rows + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        0,  # LEFT X
                        (self.grid_size + 1) * r + 1,  # Y
                        (self.grid_size + 1) * self.cols,  # RIGHT X
                        (self.grid_size + 1) * r + 1,  # Y
                    ),
                ),
                ("c3B", (*_BLACK, *_BLACK)),
            )

        # horizontal lines
        for c in range(self.cols + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * c + 1,  # X
                        0,  # BOTTOM Y
                        (self.grid_size + 1) * c + 1,  # X
                        (self.grid_size + 1) * self.rows,  # TOP X
                    ),
                ),
                ("c3B", (*_BLACK, *_BLACK)),
            )
        batch.draw()

    def _draw_resource_sight_overlay(self, env):
        sight_cells = {}
        for resource in env.resources:
            row, col = resource.position
            row_start = max(0, row - env.sight)
            row_end = min(self.rows - 1, row + env.sight)
            col_start = max(0, col - env.sight)
            col_end = min(self.cols - 1, col + env.sight)
            for r in range(row_start, row_end + 1):
                for c in range(col_start, col_end + 1):
                    sight_cells[(r, c)] = sight_cells.get((r, c), 0) + 1

        if not sight_cells:
            return

        base_alpha = 30
        alpha_step = 30
        max_alpha = 150
        batch = pyglet.graphics.Batch()
        for (row, col), overlap_count in sight_cells.items():
            alpha = min(max_alpha, base_alpha + alpha_step * (overlap_count - 1))
            x0 = (self.grid_size + 1) * col + 1
            y0 = self.height - (self.grid_size + 1) * (row + 1) + 1
            x1 = x0 + self.grid_size - 1
            y1 = y0 + self.grid_size - 1
            batch.add(
                4,
                GL_QUADS,
                None,
                ("v2f", (x0, y0, x1, y0, x1, y1, x0, y1)),
                ("c4B", (*_LIGHT_BLUE, alpha, *_LIGHT_BLUE, alpha, *_LIGHT_BLUE, alpha, *_LIGHT_BLUE, alpha)),
            )
        batch.draw()

    def _draw_tasks(self, env):
        tasks = list(env.tasks.values())
        visible_apples = []
        invisible_apples = []
        batch_visible = pyglet.graphics.Batch()
        batch_invisible = pyglet.graphics.Batch()

        for task in tasks:
            row, col = task.position
            apple = pyglet.sprite.Sprite(
                    self.img_apple,
                    (self.grid_size + 1) * col,
                    self.height - (self.grid_size + 1) * (row + 1),
                    batch=None,
                )
            apple.update(scale=self.grid_size / apple.width)
            if task.is_visible(env.current_step):
                apple.batch = batch_visible
                visible_apples.append(apple)
            else:
                apple.batch = batch_invisible
                apple.opacity = 50
                invisible_apples.append(apple)
        
        batch_visible.draw()
        batch_invisible.draw()

        for task in tasks:
            row, col = task.position
            capability_str = ",".join(map(str, np.round(task.capability, 1)))
            self._draw_capability_badge(row, col, capability_str)
            spawn_time = str(task.spawn_time)
            self._draw_time_badge(row, col, spawn_time)
            self._draw_dynamicType_badge(row, col, task.dynamic_type)

    def _draw_resources(self, env):
        resources = []
        batch = pyglet.graphics.Batch()

        for resource in env.resources:
            row, col = resource.position
            resources.append(
                pyglet.sprite.Sprite(
                    self.img_agent,
                    (self.grid_size + 1) * col,
                    self.height - (self.grid_size + 1) * (row + 1),
                    batch=batch,
                )
            )
        for r in resources:
            r.update(scale=self.grid_size / r.width)
        batch.draw()
        for r in env.resources:
            capability_str = ",".join(map(str, np.round(r.capability, 1)))
            self._draw_capability_badge(*r.position, capability_str)

    def _draw_capability_badge(self, row, col, capability): # draw capability badge
        resolution = 4
        text_length = len(str(capability))
        font_size = 10 if text_length > 3 else 12
        char_width = font_size * 0.5
        rect_width = char_width * text_length * 1.3
        rect_height = font_size * 1.5
        # calculate position of badge
        badge_x = col * (self.grid_size + 1) + (3 / 4) * (self.grid_size + 1)
        badge_y = (
            self.height
            - (self.grid_size + 1) * (row + 1)
            + (1 / 4) * (self.grid_size + 1)
        )

        # make a circle
        verts = []
        verts += [badge_x-rect_width/2, badge_y-rect_height/2]
        verts += [badge_x-rect_width/2, badge_y+rect_height/2]
        verts += [badge_x+rect_width/2, badge_y+rect_height/2]
        verts += [badge_x+rect_width/2, badge_y-rect_height/2]

        rect = pyglet.graphics.vertex_list(resolution, ("v2f", verts))
        glColor3ub(*_WHITE)
        rect.draw(GL_POLYGON)
        glColor3ub(*_BLACK)
        rect.draw(GL_LINE_LOOP)

        label = pyglet.text.Label(
            str(capability),
            font_name="Times New Roman",
            font_size=font_size,
            bold=True,
            x=badge_x + 1,
            y=badge_y,
            anchor_x="center",
            anchor_y="center",
            color=(*_BLACK, 255),
        )
        label.draw()

    def _draw_time_badge(self, row, col, time): # draw time badge
        resolution = 20

        font_size = 8 if len(str(time)) > 2 else 10
        radius = font_size * 0.8 # radius of badge
        # calculate position of badge
        badge_x = col * (self.grid_size + 1) + (3 / 4) * (self.grid_size + 1)
        badge_y = (
            self.height
            - (self.grid_size + 1) * (row + 1)
            + (3 / 4) * (self.grid_size + 1)
        )

        # make a circle
        verts = []
        for i in range(resolution):
            angle = 2 * math.pi * i / resolution
            x = radius * math.cos(angle) + badge_x
            y = radius * math.sin(angle) + badge_y
            verts += [x, y]
        circle = pyglet.graphics.vertex_list(resolution, ("v2f", verts))
        glColor3ub(*_WHITE)
        circle.draw(GL_POLYGON)
        glColor3ub(*_BLACK)
        circle.draw(GL_LINE_LOOP)

        label = pyglet.text.Label(
            str(time),
            font_name="Times New Roman",
            font_size=font_size,
            bold=True,
            x=badge_x,
            y=badge_y,
            anchor_x="center",
            anchor_y="center",
            color=(*_BLACK, 200),
        )
        label.draw()

    def _draw_dynamicType_badge(self, row, col, dynamicType):
        if dynamicType == TaskDynamicType.NONE:
            dynamicTypeText = 'N'
        elif dynamicType == TaskDynamicType.LINEAR_DECAY:
            dynamicTypeText = 'LD'
        elif dynamicType == TaskDynamicType.EXPONENTIAL_DECAY:
            dynamicTypeText = 'ED'
        elif dynamicType == TaskDynamicType.LINEAR_GROWTH:
            dynamicTypeText = 'LG'
        elif dynamicType == TaskDynamicType.EXPONENTIAL_GROWTH:
            dynamicTypeText = 'EG'
        elif dynamicType == TaskDynamicType.RANDOM_FLUCTUATE:
            dynamicTypeText = 'R'
        else:
            dynamicTypeText = '?'
            raise ValueError("Invalid dynamic type")

        badge_x = col * (self.grid_size + 1) + (1 / 5) * (self.grid_size + 1)
        badge_y = (
            self.height
            - (self.grid_size + 1) * (row + 1)
            + (4 / 5) * (self.grid_size + 1)
        )

        font_size = 10

        label = pyglet.text.Label(
            dynamicTypeText,
            font_name="Times New Roman",
            font_size=font_size,
            bold=True,
            x=badge_x,
            y=badge_y,
            anchor_x="center",
            anchor_y="center",
            color=(*_PURPLE, 200),
        )
        label.draw()
