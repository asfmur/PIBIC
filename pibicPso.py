import numpy as np
import Topology

PARTICLES = 50
ITERATIONS = 1000
C1 = C2 = 2.05
phi = C1 + C2
CHI = 2 / (phi - 2 + np.sqrt(phi**2 - 4 * phi))
BEST_FITNESS_PER_ITERATIONS = []

class Particle:
    def __init__(self, position, pbest, pbest_fitness):
        self.position = position
        self.pbest = pbest
        self.pbest_fitness = pbest_fitness


def initialize_particles(dimensions, a_init, b_init):
    return np.random.uniform(a_init, b_init, (PARTICLES, dimensions))

def pso(fitness_func, dimensions, a_init, b_init, a_pso, b_pso, topology_strategy):
    swarm = [
        Particle(
            position=np.random.uniform(a_init, b_init, dimensions),
            pbest=None,
            pbest_fitness=float("inf"),
        )
        for _ in range(PARTICLES)
    ]

    for particle in swarm:
        particle.pbest = particle.position.copy()
        particle.pbest_fitness = fitness_func(particle.position)

    global_best = min(swarm, key=lambda p: p.pbest_fitness).pbest
    global_best_fitness = min(p.pbest_fitness for p in swarm)

    velocities = np.random.uniform(-1, 1, (PARTICLES, dimensions))

    for iteration in range(ITERATIONS):
        neighbors_best = topology_strategy.update_neighbor(swarm)
        BEST_FITNESS_PER_ITERATIONS.append(global_best_fitness)

        for i, particle in enumerate(swarm):
            r1, r2 = np.random.rand(dimensions), np.random.rand(dimensions)
            velocities[i] = CHI * (
                velocities[i]
                + C1 * r1 * (particle.pbest - particle.position)
                + C2 * r2 * (neighbors_best[i].position - particle.position)
            )

            particle.position += velocities[i]


            for d in range(dimensions):
                if particle.position[d] < a_pso:
                    particle.position[d] = a_pso
                    velocities[i][d] *= -1
                elif particle.position[d] > b_pso:
                    particle.position[d] = b_pso
                    velocities[i][d] *= -1

            fitness = fitness_func(particle.position)
            if fitness < particle.pbest_fitness:
                particle.pbest = particle.position.copy()
                particle.pbest_fitness = fitness

                if fitness < global_best_fitness:
                    global_best = particle.pbest
                    global_best_fitness = fitness

    return global_best_fitness, BEST_FITNESS_PER_ITERATIONS

def sphere(x):
    return np.sum(x**2)

def rastrigin(position):
    return np.sum(position**2 - 10 * np.cos(2 * np.pi * position) + 10)

global_topology = Topology.Global()
old_global_topology = Topology.OldGlobal()
local_topology = Topology.Local()
kbest_topology = Topology.KBest()

best_fitness, best_per_iter = pso(sphere, 30, -10, 10, -100, 100, global_topology)
#best_fitness, best_per_iter= pso(rastrigin, 30, 2.56, 5.12, -5.12, 5.12, kbest_topology)
print(best_per_iter)
print(best_fitness)

