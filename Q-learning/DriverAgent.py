import numpy as np
import gymnasium as gym
from Map import MapEnv 
from collections import defaultdict
from tqdm import tqdm
from matplotlib import pyplot as plt
from gymnasium.utils.env_checker import check_env


class DriverAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
    ):
        self.env = env
        # Q-table
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def _obs_to_tuple(self, obs):
        d = obs["driver"]
        dest = obs["destination"]
        return (int(d[0]), int(d[1]), int(dest[0]), int(dest[1]))

    def choose_action(self, obs: tuple[int, int, bool]) -> int: 
        obs_tuple = self._obs_to_tuple(obs)
        # Exploration
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # Exploitation
        else :
            return int(np.argmax(self.q_values[obs_tuple]))
        
    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        obs_tuple = self._obs_to_tuple(obs)
        next_obs_tuple = self._obs_to_tuple(next_obs)
        
        # Decide next best action
        future_q_value = (not terminated) * np.max(self.q_values[next_obs_tuple])

        # Bellman Equation to update Q-value
        destination = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate?
        temporal_difference = destination - self.q_values[obs_tuple][action]

        # Update q-value in direction of error
        self.q_values[obs_tuple][action] = (
            self.q_values[obs_tuple][action] + self.lr * temporal_difference
        )

        # Track learning progress
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def test_agent(self, env, n_episodes):
        total_rewards = []
        successes = []
        episode_lengths = []

        # Disable exploration for testing
        old_epsilon = self.epsilon
        self.epsilon = 0.0

        for _ in range(n_episodes):
            obs, info = env.reset()
            episode_reward = 0
            steps = 0
            done = False

            while not done:
                action = self.choose_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1
                done = terminated or truncated

            total_rewards.append(episode_reward)
            successes.append(terminated) 
            episode_lengths.append(steps)

        # Restore original epsilon
        self.epsilon = old_epsilon

        success_rate = np.mean(successes)
        average_reward = np.mean(total_rewards)
        average_length = np.mean(episode_lengths)

        # Use tqdm.write so these lines are printed cleanly when a tqdm progress
        # bar is active (prevents them from being overwritten/hidden).
        tqdm.write(f"Test Results over {n_episodes} episodes:")
        tqdm.write(f"Success Rate: {success_rate:.1%}")
        tqdm.write(f"Average Reward: {average_reward:.3f}")
        tqdm.write(f"Average Episode Length: {average_length:.1f}")
        tqdm.write(f"Std Episode Length: {np.std(episode_lengths):.1f}")

    def visualize_testing_progess(self, n_episodes=5):
        render_env = MapEnv(render_mode="human")
        old_epsilon = self.epsilon
        self.epsilon = 0.0

        for episode in range(n_episodes):
            obs, info = render_env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            print(f"\n=== Episode {episode + 1} ===")
            print(f"Driver at: {obs['driver']}, Destination at: {obs['destination']}")
            
            while not done:
                action = self.choose_action(obs)
                obs, reward, terminated, truncated, info = render_env.step(action)
                episode_reward += reward
                steps += 1
                done = terminated or truncated
                
            print(f"Episode finished in {steps} steps")
            print(f"Total reward: {episode_reward:.2f}")
            print(f"Success: {'Yes' if terminated else 'No'}")
        
        # Restore original epsilon
        self.epsilon = old_epsilon
        render_env.close()

def get_moving_avgs(arr, window, convolution_mode):
        return np.convolve(
            np.array(arr).flatten(),
            np.ones(window),
            mode=convolution_mode
        ) / window

if __name__ == "__main__":
    n_episodes = 20000
    env = MapEnv(render_mode=None)

    agent = DriverAgent(
        env=env,
        learning_rate=0.1,
        initial_epsilon=1.0,
        epsilon_decay= 1e-5,
        final_epsilon=0.01,
        discount_factor=0.95,
    )

    # TRAINING AGENT
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            # 1. Choose action
            action = agent.choose_action(obs)

            # 2. Take action
            next_obs, reward, terminated, truncated, info = env.step(action)

            # 3. Learn from experience & update Q-values
            agent.update(obs, action, reward, terminated, next_obs)

            # 4. Move to next state
            done = terminated or truncated
            obs = next_obs
        # Take less random actions over time
        agent.decay_epsilon()


    # Run evaluation with exploration disabled
    from tqdm import tqdm as _tqdm
    agent.test_agent(env, 100)

    agent.visualize_testing_progess(5)