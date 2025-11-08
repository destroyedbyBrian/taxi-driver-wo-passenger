import time
from collections import defaultdict, deque
import numpy as np
import gymnasium as gym
from Map import MapEnv

class DriverAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
        convergence_tol: float = 1e-3,   # mean |TD| threshold
        convergence_patience: int = 50   # consecutive episodes below tol
    ):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # tracking
        self.training_error = []          # per-step |TD|
        self.episode_td_means = []        # per-episode mean |TD|
        self.convergence_tol = convergence_tol
        self.convergence_patience = convergence_patience
        self.convergence_episode = None   # first episode where rolling mean < tol for 'patience' episodes
        self.training_time_sec = 0.0

    def _obs_to_tuple(self, obs):
        d = obs["driver"]; dest = obs["destination"]
        return (int(d[0]), int(d[1]), int(dest[0]), int(dest[1]))

    def choose_action(self, obs) -> int:
        obs_tuple = self._obs_to_tuple(obs)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_values[obs_tuple]))

    def update(self, obs, action: int, reward: float, terminated: bool, next_obs):
        s = self._obs_to_tuple(obs)
        s_next = self._obs_to_tuple(next_obs)

        best_next = 0.0 if terminated else np.max(self.q_values[s_next])
        target = reward + self.discount_factor * best_next
        td = target - self.q_values[s][action]

        # Q update
        self.q_values[s][action] += self.lr * td

        # track |TD|
        self.training_error.append(abs(td))
        return abs(td)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def _check_convergence(self):
        """
        Convergence when the rolling mean of per-episode mean |TD|
        stays below self.convergence_tol for 'convergence_patience' episodes.
        """
        p = self.convergence_patience
        if len(self.episode_td_means) >= p and self.convergence_episode is None:
            window = self.episode_td_means[-p:]
            if np.mean(window) < self.convergence_tol:
                # episode indices are 1-based human-friendly
                self.convergence_episode = len(self.episode_td_means)

    def train(self, n_episodes: int):
        start = time.perf_counter()
        step_errors = deque()  # reset each episode

        for ep in range(1, n_episodes + 1):
            obs, info = self.env.reset()
            done = False
            step_errors.clear()

            while not done:
                a = self.choose_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(a)
                td_abs = self.update(obs, a, reward, terminated, next_obs)
                step_errors.append(td_abs)

                done = terminated or truncated
                obs = next_obs

            # per-episode mean |TD|
            ep_mean_td = float(np.mean(step_errors)) if len(step_errors) else 0.0
            self.episode_td_means.append(ep_mean_td)

            # epsilon schedule
            self.decay_epsilon()

            # check convergence condition
            self._check_convergence()

        self.training_time_sec = time.perf_counter() - start

    def test_agent(self, env, n_episodes: int):
        total_rewards, successes, lengths = [], [], []
        old_epsilon = self.epsilon
        self.epsilon = 0.0  # greedy eval

        for _ in range(n_episodes):
            obs, info = env.reset()
            done = False
            ep_r = 0.0
            steps = 0
            while not done:
                a = self.choose_action(obs)
                obs, r, terminated, truncated, info = env.step(a)
                ep_r += r
                steps += 1
                done = terminated or truncated
            total_rewards.append(ep_r)
            successes.append(terminated)
            lengths.append(steps)

        self.epsilon = old_epsilon

        # metrics
        success_rate = float(np.mean(successes))
        avg_reward = float(np.mean(total_rewards))
        avg_length = float(np.mean(lengths))
        std_length = float(np.std(lengths))

        # print
        print(f"Test Results over {n_episodes} episodes:")
        print(f"  Success Rate: {success_rate:.1%}")
        print(f"  Average Reward: {avg_reward:.3f}")
        print(f"  Average Episode Length: {avg_length:.1f}")
        print(f"  Std Episode Length: {std_length:.1f}")

        return {
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "avg_length": avg_length,
            "std_length": std_length,
        }


if __name__ == "__main__":
    n_episodes = 20000
    env = MapEnv(render_mode=None)

    agent = DriverAgent(
        env=env,
        learning_rate=0.1,
        initial_epsilon=1.0,
        epsilon_decay=1e-5,
        final_epsilon=0.01,
        discount_factor=0.95,
        convergence_tol=1e-3,        # tweak as needed
        convergence_patience=100     # e.g., 100 consecutive episodes
    )

    # TRAIN
    agent.train(n_episodes=n_episodes)

    # EVAL
    metrics = agent.test_agent(env, 100)

    # SUMMARY (matching your VI report fields)
    print("\n=== Summary (Q-Learning) ===")
    print(f"Optimal path length: {metrics['avg_length']:.1f}")                 # avg steps (greedy policy)
    print(f"Final Episode Reward: {metrics['avg_reward']:.3f}")                # avg reward
    print(f"Final Success Rate: {metrics['success_rate']:.1%}")                # success rate
    print(f"Episode to convergence: {agent.convergence_episode or n_episodes}")# first episode meeting criterion
    print(f"Total Computational Time: {agent.training_time_sec:.4f}s")         # wall-clock training time
