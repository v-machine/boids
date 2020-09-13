from typing import List, Tuple, Callable
from boids.boids import Boids
import tkinter, numpy as np, math

class Model:
    """Defines the model of the annimation

    Model of the MVC architecture. Initiate model objects here.
    """
    def __init__(self, environ_bounds: List[int]):
        self.boids = Boids(dims=3, num_boids=200, 
                           environ_bounds=environ_bounds,
                           max_velocity=2, max_acceleration=1,
                           perceptual_range=100)
        self.center = np.asarray(environ_bounds)//2
        self.cube = Cube(self.center, size=max(environ_bounds))

    def get_center(self) -> np.ndarray:
        return self.center

class Cube:
    """Creates Reference Cube in the 3d simulated environment
    """
    RELATIVE_POS = np.asarray(
        [[-1, -1, -1],
         [ 1, -1, -1],
         [ 1,  1, -1],
         [-1,  1, -1],
         [-1, -1,  1],
         [ 1, -1,  1],
         [ 1,  1,  1],
         [-1,  1,  1]])

    def __init__(self, center: Tuple[int, int, int]=(0, 0, 0),
                 size: int=1) -> None:
        self.center = np.asarray(center)
        self.size = size
        self.vertices = self._make_vertices()

    def get_vertices(self):
        return self.vertices

    def _make_vertices(self) -> np.ndarray:
        """Returns all points in a cube as a matrix"""
        return self.center + 0.5*self.size*Cube.RELATIVE_POS 

class View:
    """Defines the view of the annimation
    
    View of the MVC architecture. Contains view objects to be rendered.

    Attributes:
        canvas:
            tkinter canvas
        model:
            The model of the annimation
        drawables: List[View.DrawableInterface]
            A list of drawables 
    """
    def __init__(self, canvas: tkinter.Canvas, model: Model):
        self.canvas = canvas
        self.model = model
        self.drawables: List[View.DrawableInterface] = []
        self._create_drawables()
        self.rotate3d = Rotate3D(self.model.get_center())

    def _create_drawables(self):
        self.drawables.append(View.DrawBoids(self.model.boids),
                              View.DrawCube(self.model.cube))

    def redraw_all(self):
        """Clear view and redraw all view objects
        """
        self.canvas.delete(tkinter.ALL)
        for d in self.drawables:
            d.draw(self.canvas)

    def repeat(self, timestep: int, time_stepped_func: Callable):
        """Repeatly calls time_stepped_func at every time step

        Params:
            timestep:
                The size of timestep of the annimation
            time_stepped_func:
                The function to be called at each time step
        """
        self.canvas.after(timestep, time_stepped_func)

    class DrawableInterface:
        """Defines the Drwable interface. 

        Attributes:
            ref_to_model_data:
                Reference to pertinent model data to be drawn
        """
        def __init__(self, ref_to_model_data: object):
            self.data = ref_to_model_data

        def draw(self, canvas: tkinter.Canvas):
            """Draws the drawable object
            """
            raise NotImplementedError

    class DrawBoids(DrawableInterface):
        """Draws the boid in the gui
        """
        def __init__(self, boids: Boids):
            self.boids = boids

        def draw(self, canvas: tkinter.Canvas):
            self. draw_circle_boids(canvas)

        def draw_circle_boids(self, canvas: tkinter.Canvas):
            size = 40
            length = 4
            color = 'yellow'
            locations = self.boids.get_locations()[:, :2]
            velocities = self.boids.get_velocities()[:, :2]
            for loc, vel in zip(locations, velocities):
                x0, y0 = tuple(loc)
                x1, y1 = tuple(loc + vel * length)
                x2, y2 = tuple(loc - size // 2)
                x3, y3 = tuple(loc + size // 2)
                canvas.create_oval(x2, y2, x3, y3, fill=color)
                canvas.create_line(x0, y0, x1, y1)
    
    class DrawCube:
        """Draws the cube in the gui 
        """
        def __init__(self, cube: Cube):
            self.cube = cube

        def draw(self, canvas):
            width = 1
            color = "white"
            verts = self.data.get_vertices()
            n = len(verts)//2
            for i in range(n):
                x1, y1 = tuple(verts[i][:-1])
                x2, y2 = tuple(verts[(i+1) % n][:-1])
                x3, y3 = tuple(verts[i+n][:-1])
                x4, y4 = tuple(verts[(i+1) % n + n][:-1])
                for points in ((x1, y1, x2, y2),
                               (x3, y3, x4, y4),
                               (x1, y1, x3, y3)):
                    canvas.create_line(*points, width=width, fill=color)

    class Rotate3D:
        """Performs 3D rotation of gui objects
        """
        def __init__(self, center: np.ndarray):
            self.theta = 0
            self.center = center
            self.axis = 0

        def rotate_3D_points(self, vertices: np.ndarray) -> np.ndarray:
            """Rotates a matrix of points around self.center 
            along a given axis by self.theta degree"""
            rotmats = [View.Rotate3D._rotmat_yz,
                       View.Rotate3D._rotmat_xz,
                       View.Rotate3D._rotmat_xy]
            vertices -= self.center
            a = math.radians(self.theta)
            rotmat = rotmats[self.axis](self, math.sin(a), math.cos(a))
            vertices = np.dot(vertices, rotmat)
            vertices += self.center
        
        def _rotmat_xy(self, sin: float, cos: float) -> np.ndarray:
            return np.asarray(
                [[ cos, sin, 0],
                 [-sin, cos, 0],
                 [   0,   0, 1]])
        
        def _rotmat_xz(self, sin: float, cos: float) -> np.ndarray:
            return np.asarray(
                [[cos, 0, -sin],
                 [  0, 1,    0],
                 [sin, 0,  cos]])

        def _rotmat_yz(self, sin: float, cos: float) -> np.ndarray:
            return np.asarray(
                [[1,    0,   0],
                 [0,  cos, sin],
                 [0, -sin, cos]])

class Controller:
    """Defines the controller of the annimation.
    
    Controller of the MVC architecture. Contains event bindings
    and timestep functions.
    """
    def __init__(self, root: tkinter.Tk, view: View, model: Model):
        self.root = root
        self.view = view
        self.model = model
        self.timestep = 100

    def set_timestep(self, stepsize: int):
        """Sets timestep sizes
        
        Params:
            stepsize:
                The size of timestep
        """
        self.timestep = stepsize

    def start_loop(self):
        """Starts the annimation loop
        """
        self.view.redraw_all()
        self.event_binding()
        self._time_stepped_wrapper()
        self.root.mainloop()

    def event_binding(self):
        """Binds event to triggering functions
        """
        self.root.bind('<Button-1>', lambda event: self._mouse_pressed_wrapper(event))

    def key_pressed(self, event: tkinter.Event):
        """Defines key-press actions

        Params:
            event:
                tkinter event object
        """
        pass

    def mouse_pressed(self, event: tkinter.Event):
        """Defines mouse-press actions

        Params:
            event:
                tkinter event object
        """
        self.model.boids.swarm()

    def time_stepped(self):
        """Defines time based actions
        """
        self.model.boids.swarm()

    def _mouse_pressed_wrapper(self, event: tkinter.Event):
        self.mouse_pressed(event)
        self.view.redraw_all()

    def _time_stepped_wrapper(self):
        self.time_stepped()
        self.view.redraw_all()
        self.view.repeat(self.timestep, self._time_stepped_wrapper)


def main(width=1800, height=1800):
    root = tkinter.Tk()
    canvas = tkinter.Canvas(root, width=width, height=height, bd=0,
                            highlightthickness=0)
    canvas.pack()
    model = Model(width, height)
    view = View(canvas, model)
    controller = Controller(root, view, model)
    controller.set_timestep(10)
    controller.start_loop()
    print('window closed')


if __name__ == '__main__':
    main()
