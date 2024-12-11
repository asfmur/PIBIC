import numpy as np


PARTICLES = 30
ITERATIONS = 1000
C1 = C2 = 2.05
phi = C1 + C2
CHI = 2 / (phi - 2 + np.sqrt(phi**2 - 4 * phi))
BEST_FITNESS_PER_ITERATIONS = []

def initialize_particles(dimensions, a_init, b_init):
    return np.random.uniform(a_init, b_init, (PARTICLES, dimensions))

def compute_v_max(a_pso, b_pso):
    return (b_pso - a_pso) * 0.75

def pso(fitness_func, dimensions, a_init, b_init, a_pso, b_pso, 
        velocity_update_mode="constriction", inertia_weight=0.729, c1_start=2.5, c1_end=0.5, c2_start=0.5, c2_end=2.5):
    """
    Particle Swarm Optimization with selectable velocity update mode and dynamic coefficients for inertia weight mode.

    Args:
        fitness_func (function): Função a ser minimizada.
        dimensions (int): Dimensoes do espaco de busca.
        a_init (float): Limite inferior para inicializacao(prevenindo iniciar particulas no centro).
        b_init (float): Limite superior para inicializacao(prevenindo iniciar particulas no centro).
        a_pso (float): Limite inferior do espaco de busca.
        b_pso (float): Limite superior do espaco de busca.
        velocity_update_mode (str): "constriction" ou "inertia" para escolher o modo de atualizar velocity.
        inertia_weight (float): Valor para o peso da inercia".
        c1_start (float): Valor do coenficiente cognitivo inicial.
        c1_end (float): Valor do coeficiente cognitivo final.
        c2_start (float): Valor do coeficiente social inicial.
        c2_end (float): Valor do coeficiente social final.

    Returns:
        BEST_FITNESS_PER_ITERATIONS, best_fitness
    """
    position = initialize_particles(dimensions, a_init, b_init)
    velocity = np.random.uniform(-1, 1, (PARTICLES, dimensions))

    v_max = compute_v_max(a_pso, b_pso)

    p_best = position.copy()
    p_best_fitness = np.array([fitness_func(p) for p in p_best])

    neighbors_best = np.zeros_like(position)
    

    for iteration in range(ITERATIONS):
        best_particle_index = np.argmin(p_best_fitness)
        BEST_FITNESS_PER_ITERATIONS.append(p_best_fitness[best_particle_index])

        c1 = c1_start + (c1_end - c1_start) * (iteration / (ITERATIONS - 1))
        c2 = c2_start + (c2_end - c2_start) * (iteration / (ITERATIONS - 1))

        for i in range(PARTICLES):
            left = (i - 1) % PARTICLES
            right = (i + 1) % PARTICLES
            neighbors_best[i] = p_best[left] if p_best_fitness[left] < p_best_fitness[right] else p_best[right]
            
            r1, r2 = np.random.rand(dimensions), np.random.rand(dimensions)
            
            if velocity_update_mode == "constriction":
                velocity[i] = CHI * (velocity[i] + C1 * r1 * (p_best[i] - position[i]) + C2 * r2 * (neighbors_best[i] - position[i]))
            elif velocity_update_mode == "inertia":
                velocity[i] = (inertia_weight * velocity[i] + 
                               c1 * r1 * (p_best[i] - position[i]) + 
                               c2 * r2 * (neighbors_best[i] - position[i]))

            velocity[i] = np.clip(velocity[i], -v_max, v_max)

            position[i] += velocity[i]
            
            for d in range(dimensions):
                if position[i][d] < a_pso:
                    position[i][d] = a_pso
                    velocity[i][d] *= -1
                elif position[i][d] > b_pso:
                    position[i][d] = b_pso
                    velocity[i][d] *= -1
            
            current_fitness = fitness_func(position[i])
            
            if current_fitness < p_best_fitness[i]:
                p_best[i] = position[i]
                p_best_fitness[i] = current_fitness
    
    best_particle_index = np.argmin(p_best_fitness)

    return BEST_FITNESS_PER_ITERATIONS, p_best_fitness[best_particle_index]


def sphere(x):
    return np.sum(x**2)

def rastrigin(position):
    return np.sum(position**2 - 10 * np.cos(2 * np.pi * position) + 10)


#sphere_cons_best_per_iter, sphere_cons_best_fitness = pso(sphere, 30, -100, 100, -100, 100)
ras_cons_best_per_iter, ras_cons_best_fitness = pso(rastrigin, 30, 2.56, 5.12, -5.12, 5.12)


#sphere_iw_best_per_iter, sphere_iw_best_fitness= pso(sphere, 30, -100, 100, -100, 100, velocity_update_mode="inertia")
#ras_iw_best_per_iter, ras_iw_best_fitness = pso(rastrigin, 30, 2.56, 5.12, -5.12, 5.12, velocity_update_mode="inertia")
print(ras_cons_best_per_iter)
print("Best Fitness:", ras_cons_best_fitness)
