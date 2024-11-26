import numpy as np
import gymnasium as gym

def discretize_state(state, bins):
    return tuple(np.digitize(state[i], bins[i]) for i in range(len(state)))

env = gym.make("MountainCar-v0", render_mode=None)

n_bins = 20
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
episodes = 5000
max_steps = 200

bins = [
    np.linspace(-1.2, 0.6, n_bins),
    np.linspace(-0.07, 0.07, n_bins)
]

q_table_shape = tuple([n_bins] * env.observation_space.shape[0]) + (env.action_space.n,)
q_table = np.zeros(q_table_shape)

for episode in range(episodes):
    state, _ = env.reset()
    state = discretize_state(state, bins)
    done = False

    for step in range(max_steps):
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = discretize_state(next_state, bins)

        if done and next_state[0] >= bins[0][-1]:
            reward = 0
        else:
            reward = -1

        q_table[state][action] = q_table[state][action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state][action]
        )

        state = next_state

        if done:
            break

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

test_episodes = 10
for episode in range(test_episodes):
    env.close()
    env = gym.make("MountainCar-v0", render_mode="human")
    state, _ = env.reset()
    state = discretize_state(state, bins)
    done = False

    for step in range(max_steps):

        action = np.argmax(q_table[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = discretize_state(next_state, bins)
        state = next_state

        if done:
            break

env.close()
