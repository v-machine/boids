# Boids Simulation

## Purpose

The [Boids](https://www.red3d.com/cwr/boids/) was first created in 1986 by Graig Reynolds to simulate complex swarm behavior emerged from simple bahavioral rules. The exploration done here focuses primarily on the optimization aspect of the Boids simulation, namely data structures and update algorithms that enable the simulation to run smoothly as the size of the swarm increases. Other explorations include making the rules governing the boids' behaviors idiosyncratic; utilizing multi-threading (as opposed to sequentially) to more closely resemble concomitant agent actions in the real world.

## Boids Overview

Assuming that Boids are two dimensional (they don't have to be!):\
Boids attributes:

- location (x1, x2)
- speed/orientation (v1, v2)
- acceleration (a1, a2)

Three basic agent rules:

1) separation
2) alignment 
3) cohesion

Global parameters:

1) field of perception

## Optimization

The following are a few chosen method for optimization:

- vectorization
- space partitioning algorithms (KD-tree, Ball-tree)

### Vectorization

Given n boids, each with three attributes (i.e., location, speed/orientation, and acceleration) the state of the entire simulation can be represented by a 2 x n x 3 matrix: 

```
[[[b0x1, b0x2, b0v1],
  [b1x1, b1x2, b1v1], 
   ...
  [bnx1, bnx2, bnv1]],

  [b0v2, b0a1, b0a2],
  [b1v2, b1a1, b1a2],
   ...
  [bnv2, bna1, bna2]]]
```

where 2 is the number of dimension of the simulation and 3 being the attributes describing the state of each boid.

## Moonshot

- maybe this can be generalized into a particle simulation framework