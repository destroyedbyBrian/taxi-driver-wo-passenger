# === SARSA with final metrics ===
import time
from collections import defaultdict, deque
import numpy as np

# Try to import MapEnv; if in Colab and not found, prompt upload of Map.py
try:
    from Map import MapEnv
except ModuleNotFoundError:
    try:
        from google.colab import files
        print("Upload Map.py from your project...")
        files.upload()
        from Map import MapEnv
    except Exception as e:
        raise

class SarsaAgent:
    def __init__(
        self,
        env,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 1e-5,
        final_epsilon: float = 0.01,
        convergence_tol: float = 1e-3,     # threshold on mean |TD| (per-episode)
        convergence_patience: int = 100    # require this many consecutive episodes under tol
    ):
        self.env = env
        self.nA = env.action_space.n
        self.q = defaultdict(lambda: np.zeros(self.nA, dtype=float))

        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Tracking for convergence & timing
        self.episode_td_means = []           # per-episode mean |TD|
        self.convergence_tol = convergence_tol
        self.convergence_patience = convergence_patience
        self.convergence_episode = None      # first episode that meets criterion
        self.training_time_sec = 0.0

    @staticmethod
    def _obs_to_key(obs):
        d = obs["driver"]; g = obs["destination"]
        return (int(d[0]), int(d[1]), int(g[0]), int(g[1]))

    def _epsilon_greedy(self, s_key):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q[s_key]))

    def _greedy(self, s_key):
        return int(np.argmax(self.q[s_key]))

    def _check_convergence(self):
        p = self.convergence_patience
        if self.convergence_episode is None and len(self.episode_td_means) >= p:
            if np.mean(self.episode_td_means[-p:]) < self.convergence_tol:
                self.convergence_episode = len(self.episode_td_means)

    def train(self, n_episodes: int = 20000):
        start = time.perf_counter()

        for ep in range(1, n_episodes + 1):
            obs, info = self.env.reset()
            s = self._obs_to_key(obs)
            a = self._epsilon_greedy(s)

            done = False
            td_errors = []

            while not done:
                next_obs, r, terminated, truncated, info = self.env.step(a)
                s_next = self._obs_to_key(next_obs)

                # choose next action (on-policy)
                a_next = self._epsilon_greedy(s_next)

                # SARSA target: r + γ Q(s', a')
                target = r + (0.0 if (terminated or truncated) else self.gamma * self.q[s_next][a_next])
                td = target - self.q[s][a]
                self.q[s][a] += self.lr * td

                td_errors.append(abs(td))

                s, a = s_next, a_next
                done = terminated or truncated

            # per-episode |TD| mean
            ep_mean_td = float(np.mean(td_errors)) if td_errors else 0.0
            self.episode_td_means.append(ep_mean_td)

            # epsilon schedule
            self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

            # check convergence condition
            self._check_convergence()

        self.training_time_sec = time.perf_counter() - start

    def test(self, n_episodes: int = 100, max_eval_steps: int = 200):
        # Evaluate with greedy policy (ε=0)
        old_eps = self.epsilon
        self.epsilon = 0.0

        total_rewards, successes, lengths = [], [], []

        for _ in range(n_episodes):
            obs, info = self.env.reset()
            s = self._obs_to_key(obs)
            done = False
            ep_r = 0.0
            steps = 0

            while not done and steps < max_eval_steps:
                a = self._greedy(s)
                obs, r, terminated, truncated, info = self.env.step(a)
                ep_r += r
                steps += 1
                s = self._obs_to_key(obs)
                done = terminated or truncated

            total_rewards.append(ep_r)
            successes.append(terminated)
            lengths.append(steps)

        self.epsilon = old_eps

        # Metrics
        success_rate = float(np.mean(successes))
        avg_reward   = float(np.mean(total_rewards))
        avg_length   = float(np.mean(lengths))
        std_length   = float(np.std(lengths))

        return {
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "avg_length": avg_length,
            "std_length": std_length,
        }

if __name__ == "__main__":
    # Hyperparameters (match your other experiments where possible)
    n_episodes = 20000
    env = MapEnv(render_mode=None)

    agent = SarsaAgent(
        env=env,
        learning_rate=0.1,
        discount_factor=0.95,
        initial_epsilon=1.0,
        epsilon_decay=1e-5,
        final_epsilon=0.01,
        convergence_tol=1e-3,
        convergence_patience=100
    )

    # TRAIN
    agent.train(n_episodes=n_episodes)

    # TEST
    metrics = agent.test(n_episodes=100)

    # PRINT SUMMARY (aligns with your VI & QL reports)
    print("\n=== Summary (SARSA) ===")
    print(f"Optimal path length: {metrics['avg_length']:.1f}")                 # avg steps under greedy policy
    print(f"Final Episode Reward: {metrics['avg_reward']:.3f}")                # avg reward over test episodes
    print(f"Final Success Rate: {metrics['success_rate']:.1%}")                # success rate
    print(f"Episode to convergence: {agent.convergence_episode or n_episodes}")# first ep meeting tol for patience
    print(f"Total Computational Time: {agent.training_time_sec:.4f}s")         # wall-clock training time
