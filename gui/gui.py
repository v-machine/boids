from typing import List, Tuple, Dict, Callable
from boids.boids import Boids
import tkinter, numpy as np, math

class Model:
    """Defines the model of the annimation

    Model of the MVC architecture. Initiate model objects here.
    """
    def __init__(self, center: List[int], environ_bounds: List[int]):
        self.center = center
        self.boids = Boids(num_boids=200, environ_bounds=environ_bounds,
                           max_velocity=2, max_acceleration=1,
                           perceptual_range=100, origin=center)
        self.cube = Cube(self.center, size=max(environ_bounds))

    def get_center(self) -> np.ndarray:
        return np.asarray(self.center)

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

    def __init__(self, model_center: List[int],
                 size: int=1) -> None:
        self.center = np.asarray(model_center)
        self.size = size
        self.vertices = self._make_vertices()

    def get_vertices(self):
        return self.vertices.copy()

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
    def __init__(self, canvas, model):
        self.canvas = canvas
        self.model = model
        self.center = model.get_center()
        self.basis = np.identity(self.center.size)
        self.drawables = self._create_drawables()
    
    def _create_drawables(self):
        return [DrawCube(self, self.model.cube),
                DrawBoids(self, self.model.boids)]
    
    def get_center(self):
        return self.center.copy()

    def project(self, vectors) -> np.ndarray:
        return View.Util.project(vectors, self.basis, self.center)

    def rotate_bais(self, axis: int, angle: int):
        self.basis = View.Util.rotate3d(self.basis, np.zeros(3),
                                        axis, angle)

    def redraw_all(self):
        self.canvas.delete(tkinter.ALL)
        for d in self.drawables:
            d.draw_wrapper(self.canvas)

    def repeat(self, timestep, time_stepped_func):
        self.canvas.after(timestep, time_stepped_func)

    class Util:
        """Stores utility functions for view
        """
        @staticmethod
        def project(vectors, basis, center) -> np.ndarray:
            """Projects vectors to the given basis and center
            """
            return np.dot((vectors-center), basis) + center

        @staticmethod
        def rotate3d(vectors: np.ndarray, center: np.ndarray,
                   axis: int, angle: int) -> np.ndarray:
            """Rotates a vector around a given center, axis, and degree"""
            rotmats = [View.Util._rotmat_yz,
                       View.Util._rotmat_xz,
                       View.Util._rotmat_xy]
            a = math.radians(angle+angle)
            rotmat = rotmats[axis](math.sin(a), math.cos(a))
            return np.dot((vectors-center), rotmat) - center
        
        @staticmethod
        def _rotmat_xy(sin: float, cos: float) -> np.ndarray:
            return np.asarray(
                [[cos, -sin, 0],
                 [sin,  cos, 0],
                 [  0,    0, 1]])
        
        @staticmethod
        def _rotmat_xz(sin: float, cos: float) -> np.ndarray:
            return np.asarray(
                [[ cos, 0, sin],
                 [   0, 1,   0],
                 [-sin, 0, cos]])
        
        @staticmethod
        def _rotmat_yz(sin: float, cos: float) -> np.ndarray:
            return np.asarray(
                [[1,   0,    0],
                 [0, cos, -sin],
                 [0, sin,  cos]])

class DrawableInterface:
    """Defines the Drwable interface. 

    Attributes:
        ref_to_model_data:
            Reference to pertinent model data to be drawn
    """
    def __init__(self, view: View, model):
        self.view = view
        self.model = model
        self.data = self.get_data()

    def get_data(self) -> Dict[str, np.ndarray]: 
        """Returns the data to be drawn from model object
        """
        raise NotImplementedError

    def draw(self, canvas: tkinter.Canvas):
        """Draws the drawable object
        """
        raise NotImplementedError

    def draw_wrapper(self, canvas: tkinter.Canvas):
        self.data = self.get_data()
        for item in self.data:
            self.data[item] = self.view.project(self.data[item])
        self.draw(canvas)

class DrawCube(DrawableInterface):
    """Draws the cube in the gui 
    """
    def __init__(self, view: View, cube: Cube):
        super().__init__(view=view, model=cube)

    def get_data(self) -> Dict[str, np.ndarray]:
        return {"vertices": self.model.get_vertices()}

    def draw(self, canvas):
        width = 5
        color = "white"
        verts = self.data["vertices"]
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

class DrawBoids(DrawableInterface):
    """Draws the boid in the gui
    """
    def __init__(self, view: View, boids: Boids):
        super().__init__(view=view, model=boids)
        
    def get_data(self) -> Dict[str, np.ndarray]:
        data = {"locations": self.model.get_locations(),
                "velocities": self.model.get_velocities()}
        # self._center_in_view(data)
        return data

    def _center_in_view(self, data: Dict[str, np.ndarray]):
        move = self.view.get_center() - self.model.get_env_bounds()//2
        for item in data:
            data[item] += move
    
    def draw(self, canvas: tkinter.Canvas):
        self.draw_circle_boids(canvas)
     
    def draw_circle_boids(self, canvas: tkinter.Canvas):
        size = 40
        length = 4
        color = 'yellow'
        locations = self.data["locations"][:, :2]
        velocities = self.data["velocities"][:, :2]
        for loc, vel in zip(locations, velocities):
            x0, y0 = tuple(loc)
            x1, y1 = tuple(loc + vel * length)
            x2, y2 = tuple(loc - size // 2)
            x3, y3 = tuple(loc + size // 2)
            canvas.create_oval(x2, y2, x3, y3, fill=color)
            # canvas.create_line(x0, y0, x1, y1)

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
        self.root.bind('<Key>', lambda event: self._key_pressed_wrapper(event))

    def key_pressed(self, event):
        """Defines key-press actions

        Params:
            event:
                tkinter event object
        """
        degree = 1
        key_to_args = {
            "Up": (0, degree),
            "Down": (0, -degree),
            "Left": (1, degree),
            "Right": (1, -degree),
            "comma": (2, degree),
            "period": (2, -degree)}
        if event.keysym in key_to_args:
            self.view.rotate_bais(*key_to_args[event.keysym])

    def mouse_pressed(self, event):
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

    def _key_pressed_wrapper(self, event: tkinter.Event):
        self.key_pressed(event)
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
    model = Model(center=[width//2, height//2, height//2],
                  environ_bounds=[1000, 1000, 1000])
    view = View(canvas, model)
    controller = Controller(root, view, model)
    controller.set_timestep(10)
    controller.start_loop()
    print('window closed')


if __name__ == '__main__':
    main()
