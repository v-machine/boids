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
        return View.Util.project(vectors, self.basis, self.center,
                                 self.center, distance=3000,
                                 focal_length=2000)

    def rotate_basis(self, axis: int, angle: int):
        self.basis = View.Util.rotate_basis_3d(self.basis, axis, angle)

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
        def project(vectors, basis, rot_center, camera_loc, 
                    distance, focal_length) -> np.ndarray:
            """Projects vectors to the given basis and center
            """
            rotated = View.Util.rotate(vectors, basis, rot_center)
            # return View.Util._perspective_project(rotated, basis, camera_loc, 
                                                  # distance, focal_length)
            return View.Util._orthographic_project(rotated, basis, camera_loc)

        @staticmethod
        def _orthographic_project(vectors: np.ndarray, basis: np.ndarray,
                                  camera_loc: np.ndarray) -> np.ndarray:
            """Project vectors in an orthographic view
            """
            return (np.dot((vectors-camera_loc), basis) +
                    camera_loc)[:, :-1]

        @staticmethod
        def _perspective_project(vectors: np.ndarray, basis: np.ndarray, 
                                 camera_loc: int, distance: int, 
                                 focal_length: int) -> np.ndarray:
            """Project vectors in a perspective view
            """
            z = focal_length / (distance - vectors[:, -1])
            projection_mat = np.outer(z, np.asarray([1, 1, 0]))
            return (np.multiply(vectors-camera_loc, projection_mat) + 
                    camera_loc)[:, :-1]
        
        @staticmethod
        def rotate(vectors: np.ndarray, basis: np.ndarray, 
                   rot_center: np.ndarray) -> np.ndarray:
            """Rotates vectors based on view basis
            """
            return np.dot((vectors-rot_center), basis) + rot_center

        @staticmethod
        def rotate_basis_3d(basis: np.ndarray, axis: int,
                            angle: int) -> np.ndarray:
            """Rotates a basis vector around a given axis and degree
            """
            rotmats = [View.Util._rotmat_yz,
                       View.Util._rotmat_xz,
                       View.Util._rotmat_xy]
            a = math.radians(angle+angle)
            rotmat = rotmats[axis](math.sin(a), math.cos(a))
            return np.dot(basis, rotmat)

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
        width = 4
        color = "white"
        verts = self.data["vertices"]
        n = len(verts)//2
        for i in range(n):
            x1, y1 = tuple(verts[i])
            x2, y2 = tuple(verts[(i+1) % n])
            x3, y3 = tuple(verts[i+n])
            x4, y4 = tuple(verts[(i+1) % n + n])
            for points in ((x1, y1, x2, y2),
                           (x3, y3, x4, y4),
                           (x1, y1, x3, y3)):
                canvas.create_line(*points, width=width, 
                                   fill=color, dash=(8, 8))

class DrawBoids(DrawableInterface):
    """Draws the boid in the gui
    """
    def __init__(self, view: View, boids: Boids):
        self.size = 20
        self.length = 20
        super().__init__(view=view, model=boids)
        
    def get_data(self) -> Dict[str, np.ndarray]:
        data = {"locations": self.model.get_locations(),
                "velocities": self.model.get_locations()+
                self.length*self.model.get_velocities()}
        return data

    def draw(self, canvas: tkinter.Canvas):
        self.draw_circle_boids(canvas)
     
    def draw_circle_boids(self, canvas: tkinter.Canvas):
        color = 'black'
        locations = self.data["locations"]
        velocities = self.data["velocities"]
        for loc, vel in zip(locations, velocities):
            x0, y0 = tuple(loc)
            x1, y1 = tuple(vel)
            x2, y2 = tuple((loc - self.size//2))
            x3, y3 = tuple((loc + self.size//2))
            canvas.create_oval(x2, y2, x3, y3, fill=color)
            canvas.create_line(x0, y0, x1, y1)
    
    def draw_triangle_boids(self, canvas: tkinter.Canvas):
        size = 40
        color = 'yellow'
        locations = self.data["locations"]
        velocities = self.data["velocities"]
        for loc, vel in zip(locations, velocities):
            x0, y0 = tuple(loc)
            x1, y1 = tuple((loc + vel))
            x2, y2 = tuple(loc - size // 2)
            x3, y3 = tuple(loc + size // 2)
            canvas.create_oval(x2, y2, x3, y3, fill=color)
            canvas.create_line(x0, y0, x1, y1)

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
        degree = 0.2
        key_to_args = {
            "Up": (0, degree),
            "Down": (0, -degree),
            "Left": (1, degree),
            "Right": (1, -degree),
            "comma": (2, degree),
            "period": (2, -degree)}
        if event.keysym in key_to_args:
            self.view.rotate_basis(*key_to_args[event.keysym])

    def mouse_pressed(self, event):
        """Defines mouse-press actions

        Params:
            event:
                tkinter event object
        """
        pass

    def time_stepped(self):
        """Defines time based actions
        """
        self.model.boids.swarm()
        self.auto_rotate_view()

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

    def auto_rotate_view(self):
        degree = 0.01
        for axis in range(3):
            self.view.rotate_basis(axis, degree)

def main(width=2000, height=2000):
    root = tkinter.Tk()
    canvas = tkinter.Canvas(root, width=width, height=height, bd=0,
                            highlightthickness=0)
    canvas.pack()
    model = Model(center=[width//2, height//2, height//2],
                  environ_bounds=[1200, 1200, 1200])
    view = View(canvas, model)
    controller = Controller(root, view, model)
    controller.set_timestep(10)
    controller.start_loop()
    print('window closed')


if __name__ == '__main__':
    main()
