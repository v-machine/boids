#!/usr/bin/env python3

import numpy as np
from typing import List, Tuple, Mapping, Callable, Union
from enum import IntEnum, unique

"""The Boids module contains a single class Boids.

The class Boids can be used to initialize a swarm of boids to be simulated.
The boids can be generalized to a similuation of arbitrary dimensions, and
supports the extension of hehavioral rules.

author: Vincent Mai
version: 0.1.0
"""

EPSILON = 10**-8  # prevents divison by zero

class Boids:
    """Defines a collections of boids to be simulated.

    The class Boids follows the behavioral rules as specified in 
    Craig Reynolds'1987 paper "Flocks, herds and schools"[1], namely,
    alignment, separation, and cohesion. Boids supports generalizations
    to arbitrary dimension (as much as your computing power would allow).
    It also supports extensio ntdexisting behavioral rules and fine
    tuning of how much each rule affects the boids' behaviors.

    Attributes:
        dims: int
            The number of dimension of the simulation.
        num_boids: int
            The number of boids to be simulated.
        env_dims: List[int]
            The size in each dimension of the simulated environment.
        max_vel: float
            The maximum velocity for all boids.
        max_acc: float
            The maximum accleration for all boids.
        p_range: int
            The boids perceptual range.
        rules: Mapping[Callable[np.ndarray], float]
            A mapping of the default set of behavioral rules as specified in 
            Reynolds's paper: align, separate, cohere, each to their
            coefficient, denoting its proportion of impact. In default init, 
            each rule has a default attributes of 1 / (number of rules).

    Reference:
        .. [1] Reynolds, Craig (1987). Flocks, herds and schools: A distributed
        behavioral model. SIGGRAPH '87: Proceedings of the 14th Annual
        Conference on Computer Graphics and Interactive Techniques. Association
        for Computing Machinery. pp. 25–34.       
    """
    @unique
    class Attr(IntEnum):
        """Assigns a unique index to each attributes of boids' states"""
        LOC = 0
        VEL = 1
        ACC = 2

    def __init__(self, num_boids: int,
                 environ_bounds: List[int],
                 max_velocity: float,
                 max_acceleration: float,
                 perceptual_range: int,
                 origin: List[int]=None):
        self.num_boids = num_boids
        self.env_dims = np.asarray(environ_bounds)
        self.max_vel = max_velocity
        self.max_acc = max_acceleration
        self.p_range = perceptual_range
        self.loc_diff = None
        self.rules = {Boids.align: 0,
                      Boids.separate: 1,
                      Boids.cohere: 0}
        self.origin = (self.env_dims//2 if origin is None 
                       else np.asarray(origin))
        self.state = self._init_boids_state()
        self.distance = self._compute_distance()

    def _init_boids_state(self) -> np.ndarray:
        """Initializes the state of the boids as a tensor.
        
        The state of the boids stores their attributes across dimensions,
        such as locations, velocities, and acclerations.

        Returns:
            state: np.ndarray, shape=(d, n, k)
                The state of the boids, with d,n,k being the instance attributes
                of dims, num_boids, and the number of state attributes. 

        Notes:
            The default initial state is randomized with the bounded range of
            each attributes. For instance, a random location (x1, x2...) bounded
            by the size of the simulation environment (env_dims).
        """
        dims, n_boids, n_attrs = (self.env_dims.size, 
                                  self.num_boids, len(Boids.Attr))
        max_vel, max_acc = self.max_vel, self.max_acc
        state = np.zeros([dims, n_boids, n_attrs], dtype="float")

        upper = self.origin + self.env_dims//2
        lower = self.origin - self.env_dims//2
        for idx, (high, low) in enumerate(zip(upper, lower)):
            state[idx, :, Boids.Attr.LOC] = np.random.randint(low=low, high=high,
                                                              size=n_boids)
        state[:, :, Boids.Attr.VEL] = np.random.uniform(low=-max_vel, high=max_vel,
                                                        size=(dims, n_boids))
        state[:, :, Boids.Attr.ACC] = np.random.uniform(low=-max_acc, high=max_acc,
                                                        size=(dims, n_boids))
        return state
    
    def get_locations(self) -> np.ndarray:
        """Returns the locations of the boids
        
        Returns:
            np.ndarray, shape=(n, d)
                The location of the boids, with d,n being the instance attributes
                of dims, num_boids.
        """
        return np.transpose(self.state[:, :, Boids.Attr.LOC])

    def get_velocities(self) -> np.ndarray:
        """Returns the velocitiess of the boids
        
        Returns:
            np.ndarray, shape=(n, d)
                The location of the boids, with d,n being the instance attributes
                of dims, num_boids.
        """
        return np.transpose(self.state[:, :, Boids.Attr.VEL])

    def _perceive(self) -> np.ndarray:
        """Returns a tensor storing the mutual influences between boids. 
        
        The mutual influence tensor denotes the coefficients (range[0,1]) of 
        the influences each boid receives from all other boids with its p_range.
        The coefficient is the same amongst all of a boid's neighbors, and
        zeros for those falling outside the p_range. For each boid, their
        coefficients boid should sum to 1.
        
        Returns:
            np.ndarray, shape=(n, n)
                The mutual influence as a n*n matrix; n being the 
                num_boids. 
        
        Note:
            A mutual influence exists between two boids if they are within one
            another's perceptual range. A neighborless boid will be maintiain
            its current state (i.e, coefficient=1 at its own index). 
        """
        dist = self.distance
        p_filter = np.where((dist<self.p_range) & (dist>0), 1, 0)
        num_neighbors = np.sum(p_filter, axis=-1, keepdims=True)
        mut_influence = p_filter / np.where(num_neighbors==0, 1, num_neighbors)
        diags = np.transpose(np.where(num_neighbors==0, 1, 0))
        np.fill_diagonal(mut_influence, diags)
        return mut_influence
    
    def _compute_distance(self) -> np.ndarray:
        """Returns a matrix of the square of the euclidean distance between the
        the boid at the current index and all other boids

        Returns:
            np.ndarray
                distance matrix between boids
        """
        loc = np.expand_dims(self.state[:, :, Boids.Attr.LOC], axis=-1)
        m = np.tile(loc, (1, 1, self.num_boids))
        self.loc_diff = m-m.transpose(0, 2, 1)
        return np.linalg.norm(self.loc_diff, axis=0)

    def align(self, mut_influence: np.ndarray) -> np.ndarray:
        """Returns an acceleration delta as the result of the alignment rule
        
        Algn returns an acceleration delta defined by the difference between 
        a boids' orientation and a linear combination of its neighbors' 
        orientations.
        
        Params:
            mut_influence: np.ndarray, shape=(n, n)
                The mutual influence as a n*n matrix; n being the num_boids. 
        
        Returns:
            np.ndarray, shape=(d, n)
                The acceleration delta as the difference between each boids'
                current orientation and the weighted average of its neighbors'.

        Note:
            The mut_influence tensor should be computed based on the default
            p_range. The alignment rule only steers the boids toward the average
            orientation of their neighbors. It does not change the magnitude
            of their current velocity.
        """
        vel = self.state[:, :, Boids.Attr.VEL]
        vel_norm = np.linalg.norm(vel, axis=0)
        orientation = vel / (vel_norm + EPSILON)
        desired_orientation = np.dot(orientation, mut_influence)
        desired_orientation = np.multiply(desired_orientation, 
                                          vel_norm + EPSILON)
        return desired_orientation - orientation
    
    def separate(self, mut_influence: np.ndarray) -> np.ndarray:
        """Returns an acceleration delta as the result of the separation rule
        
        The acceleration delta returned is defined by subtracting a boids'
        location and a linear combination of its neighbors' location.
        
        Params:
            mut_influence: np.ndarray, shape=(n, n)
                The mutual influence as a n*n matrix; n being the num_boids. 

        Returns:
            np.ndarray, shape=(d, n)
                The acceleration delta as vectors pointing from the boid's 
                neighbors' average location to the boid's current location.

        Note: 
            The mut_influence tensor should be generated using the lower bound
            of the proxim_bounds.
        """
        scaled_pos_diff = self.loc_diff / (self.distance + EPSILON)
        return np.einsum("ijk, jk -> ij", scaled_pos_diff, mut_influence)
        
    def cohere(self, mut_influence: np.ndarray) -> np.ndarray:
        """Returns an acceleration delta as the result of the cohesion rule
        
        The acceleration delta returned is defined by subtracting the linear 
        combination of a boid's neighbors' locations by its own location. 
        
        Params:
            mut_influence: np.ndarray, shape=(n, n)
                The mutual influence as a n*n matrix; n being the num_boids. 

        Returns:
            np.ndarray, shape=(d, n)
                The acceleration delta as vectors point from boid's current 
                location to the average locations of its neighbors.

        Note: 
            The mut_influence tensor should be generated using the lower bound
            of the proxim_bounds.
        """
        loc = self.state[:, :, Boids.Attr.LOC]
        return np.dot(loc, mut_influence) - loc

    def _update_acc(self, acc_delta: np.ndarray, coeff: float):
        """Updates the boids' accleration by an acceleration delta.
        
        Params:
            acc_delta: np.ndarray, shape=(d, n)
                The accleration to add to existing acceleration.
            coeff: float
                The update coefficient of the delta as pertained to the behavioral
                rule.
        """
        self.state[:, :, Boids.Attr.ACC] += maglim(acc_delta, 
                                                   self.max_acc*coeff)
    
    def _update_vel(self):
        """Update the boids' velocity by their accelerations."""
        self.state[:, :, Boids.Attr.VEL] += self.state[:, :, Boids.Attr.ACC]
        self.state[:, :, Boids.Attr.VEL] = maglim(
            self.state[:, :, Boids.Attr.VEL], self.max_vel)

    def _update_loc(self):
        """Update the boids' location by their velocity"""
        lower = np.expand_dims(self.origin - self.env_dims//2, -1)
        self.state[:, :, Boids.Attr.LOC] += self.state[:, :, Boids.Attr.VEL]
        self.state[:, :, Boids.Attr.LOC] -= lower
        self.state[:, :, Boids.Attr.LOC] %= np.expand_dims(self.env_dims, axis=1)
        self.state[:, :, Boids.Attr.LOC] += lower
    
    def _update_acc_by_rules(self, mut_influence: np.ndarray):
        """Updates the boids' accelerations by all of its behavioral rules.
        
        The effect of each rule is filtered by the mutual influence matrix
        and weighted by its own coefficient.
        
        Params:
            mut_influence: np.ndarray, shape=(n, n)
                The mutual influence as a n*n matrix; n being the num_boids. 
        """
        for rule, coeff in self.rules.items():
            acc_delta = rule(self, mut_influence)  # can't call self.rule
            self._update_acc(acc_delta, coeff)

    def append_rules(self, *args: Tuple[Callable, float]):
        """Append a new rule to the boids' default behavioral rules
        
        appends a new rule to the boids' existing rule
        Params:
            args: Tuple[Callable[self, np.ndarray], float]
                A variable number of tuple of functions and their corresponding
                coefficients. The behavioral function should take return value
                of self._perceive as its argument.
        Note:
            Rules must contain 'self' as their first argument
        """
        for rule, _ in args:
            setattr(Boids, rule.__name__, rule)
        self.rules.update({rule: coeff for (rule, coeff) in args})

    def _update_coeff(self, **kwargs: float):
        """Updates coefficient of rules
        
        Params:
            kwargs: rule_name = coeff
                A vriable list of rule names and their corresponding
                coefficients
        Raises:
            ValueError
                If the rule_name is not in self.rules
        """
        for rule_name, coeff in kwargs.items():
            if rule_name not in self.rules:
                raise ValueError(f"Behavioral rule {rule_name} does not exist")
            else:
                self.rules[getattr(self, rule_name)] = coeff

    def swarm(self):
        """Performs a single update on the state of the boids."""
        self.state[:, :, Boids.Attr.ACC] *= 0
        self.distance = self._compute_distance()
        mut_influence = self._perceive()
        self._update_acc_by_rules(mut_influence)
        self._update_vel()
        self._update_loc()

# Utility functions
def maglim(arr: np.ndarray, limit: float) -> np.ndarray:
    """Limits the magnitude of the input matrix/vector along the 0th axis.
    
    This function will only activate if the magnitude of the input is greater
    than the specified limit.
    
    Params:
        arr: np.ndarray, shape=(i, j) or shape=(i, )
            The input matrix or vector.
        limit: float
            The limit on the magnitude of the matrix/vetor
    
    Returns:
        np.ndarray
            The matrix/vector with a magnitude smaller than or equal to the
            limit specified.
    """
    norm = np.linalg.norm(arr, axis=0)
    return arr * np.where(norm>limit, limit / (norm+EPSILON), 1)
