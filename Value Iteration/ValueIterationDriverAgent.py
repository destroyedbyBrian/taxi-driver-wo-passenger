# Value Iteration/DriverAgent.py
import numpy as np
from Map import MapEnv

GAMMA = 0.95
THRESHOLD = 1e-4
EPSILON = 0.05
MAX_EVAL_STEPS = 200

ACTIONS = {
    0: np.array([1, 0]),   # right
    1: np.array([0, 1]),   # up
    2: np.array([-1, 0]),  # left
    3: np.array([0, -1]),  # down
}


class ValueIterationAgent:
    def __init__(self, env: MapEnv, gamma=GAMMA, threshold=THRESHOLD, epsilon=EPSILON):
        self.env = env
        self.gamma = gamma
        self.threshold = threshold
        self.epsilon = epsilon
        self.size = env.size
        self.obstacles = {tuple(obst) for obst in env._obstacles}
        self.valid_states = self._build_state_space()
        self.state_to_index = {state: idx for idx, state in enumerate(self.valid_states)}
        self.values = np.zeros(len(self.valid_states), dtype=float)
        self.delta_history: list[float] = []

    def _build_state_space(self):
        coords = [(x, y) for x in range(self.size) for y in range(self.size)]
        valid_states = []
        for driver in coords:
            if driver in self.obstacles:
                continue
            for dest in coords:
                if dest in self.obstacles or dest == driver:
                    continue
                valid_states.append((driver, dest))
        return valid_states

    def _state_index(self, driver, destination):
        return self.state_to_index[(tuple(driver), tuple(destination))]

    def _simulate_step(self, driver, destination, action):
        direction = ACTIONS[action]
        next_driver = np.clip(np.array(driver) + direction, 0, self.size - 1)
        next_driver_tuple = tuple(next_driver)
        hit_obstacle = next_driver_tuple in self.obstacles
        if hit_obstacle:
            next_driver_tuple = tuple(driver)  # blocked movement keeps position

        terminated = next_driver_tuple == tuple(destination)
        if terminated:
            reward = 100.0
            done = True
        else:
            distance = np.linalg.norm(np.array(next_driver_tuple) - np.array(destination))
            reward = -0.01 * distance
            if hit_obstacle:
                reward -= 10.0
            done = False
        return (next_driver_tuple, tuple(destination)), reward, done

    def train(self):
        delta = float("inf")
        iteration = 0
        while delta > self.threshold:
            delta = 0.0
            for idx, (driver, destination) in enumerate(self.valid_states):
                old_value = self.values[idx]
                q_values = []
                for action in range(self.env.action_space.n):
                    (next_driver, next_dest), reward, done = self._simulate_step(driver, destination, action)
                    future_value = 0.0
                    if not done:
                        next_idx = self._state_index(next_driver, next_dest)
                        future_value = self.values[next_idx]
                    q_values.append(reward + self.gamma * future_value)
                best_value = max(q_values)
                self.values[idx] = best_value
                delta = max(delta, abs(old_value - best_value))
            self.delta_history.append(delta)
            iteration += 1
        print(f"Value iteration converged in {iteration} sweeps (final delta={delta:.2e}).")

    def _state_from_obs(self, obs):
        driver = tuple(int(x) for x in obs["driver"])
        destination = tuple(int(x) for x in obs["destination"])
        return driver, destination

    def greedy_action(self, obs):
        driver, destination = self._state_from_obs(obs)
        q_values = []
        for action in range(self.env.action_space.n):
            (next_driver, next_dest), reward, done = self._simulate_step(driver, destination, action)
            future_value = 0.0
            if not done:
                next_idx = self._state_index(next_driver, next_dest)
                future_value = self.values[next_idx]
            q_values.append(reward + self.gamma * future_value)
        return int(np.argmax(q_values))

    def epsilon_greedy(self, obs):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return self.greedy_action(obs)

    def choose_action(self, obs):
        return self.epsilon_greedy(obs)

    def test_agent(self, env, n_episodes):
        total_rewards = []
        successes = []
        episode_lengths = []

        old_epsilon = self.epsilon
        self.epsilon = 0.0

        for _ in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0.0
            steps = 0
            while not done:
                action = self.greedy_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1
                done = terminated or truncated
            total_rewards.append(episode_reward)
            successes.append(terminated)
            episode_lengths.append(steps)

        self.epsilon = old_epsilon

        success_rate = np.mean(successes)
        average_reward = np.mean(total_rewards)
        average_length = np.mean(episode_lengths)

        print(f"Test Results over {n_episodes} episodes:")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Average Reward: {average_reward:.3f}")
        print(f"Average Episode Length: {average_length:.1f}")
        print(f"Std Episode Length: {np.std(episode_lengths):.1f}")

    def visualize_testing_progess(self, n_episodes=5, explore=False):
        render_env = MapEnv(render_mode="human")
        old_epsilon = self.epsilon
        if not explore:
            self.epsilon = 0.0

        try:
            for episode in range(1, n_episodes + 1):
                obs, info = render_env.reset()
                done = False
                steps = 0
                total_reward = 0.0
                print(f"\n=== Episode {episode} ===")
                print(f"Driver at: {obs['driver']}, Destination at: {obs['destination']}")

                while not done and steps < MAX_EVAL_STEPS:
                    render_env.render()
                    action = self.choose_action(obs)
                    obs, reward, terminated, truncated, info = render_env.step(action)
                    total_reward += reward
                    steps += 1
                    done = terminated or truncated

                print(f"Episode finished in {steps} steps")
                print(f"Total reward: {total_reward:.2f}")
                print(f"Success: {'Yes' if terminated else 'No'}")
        finally:
            self.epsilon = old_epsilon
            render_env.close()


if __name__ == "__main__":
    training_env = MapEnv(render_mode=None)
    agent = ValueIterationAgent(training_env)
    agent.train()
    agent.test_agent(training_env, 100)
    agent.visualize_testing_progess(10)
    training_env.close()
