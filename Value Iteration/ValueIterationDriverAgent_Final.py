# Value Iteration/DriverAgent_Final.py
import numpy as np
from matplotlib import pyplot as plt
from Map import MapEnv
from time import perf_counter

GAMMA = 0.95  # Discount factor
THRESHOLD = 1e-4  # Bellman residual threshold for convergence
EPSILON = 0.05  # Exploration rate for epsilon-greedy policy
MAX_EVAL_STEPS = 200  # Max steps per episode during evaluation

ACTIONS = {
    0: np.array([1, 0]),   # Right
    1: np.array([0, 1]),   # Up
    2: np.array([-1, 0]),  # Left
    3: np.array([0, -1]),  # Down
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
        self.training_sweeps = 0
        self.training_time = 0.0
        self.training_final_delta = float("inf")

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
        start_time = perf_counter()
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
        self.training_sweeps = iteration
        self.training_time = perf_counter() - start_time
        self.training_final_delta = delta
        print(
            f"Value iteration converged in {iteration} sweeps "
            f"(final delta={delta:.2e}, runtime={self.training_time:.3f}s)."
        )

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

    def _collect_episode_data(self, env, n_episodes, max_steps=MAX_EVAL_STEPS):
        rewards = []
        successes = []
        lengths = []
        min_steps = []
        old_epsilon = self.epsilon
        self.epsilon = 0.0
        try:
            for _ in range(n_episodes):
                obs, info = env.reset()
                start_driver = obs["driver"].copy()
                start_dest = obs["destination"].copy()
                done = False
                episode_reward = 0.0
                steps = 0
                terminated_flag = False
                while not done and steps < max_steps:
                    action = self.greedy_action(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    steps += 1
                    terminated_flag = bool(terminated)
                    done = terminated_flag or bool(truncated)
                rewards.append(episode_reward)
                successes.append(1.0 if terminated_flag else 0.0)
                lengths.append(steps)
                manhattan = np.sum(np.abs(start_driver - start_dest))
                min_steps.append(float(manhattan))
        finally:
            self.epsilon = old_epsilon
        return (
            np.array(rewards, dtype=float),
            np.array(successes, dtype=float),
            np.array(lengths, dtype=float),
            np.array(min_steps, dtype=float),
        )

    def evaluate_policy(self, env, n_episodes=200):
        return self._collect_episode_data(env, n_episodes)

    def test_agent(self, env, n_episodes=100, max_steps=MAX_EVAL_STEPS):
        rewards, successes, lengths, min_steps = self._collect_episode_data(env, n_episodes, max_steps)
        success_rate = float(np.mean(successes))
        avg_reward = float(np.mean(rewards))
        avg_length = float(np.mean(lengths))
        std_length = float(np.std(lengths))

        print(f"Test Results over {n_episodes} episodes:")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Average Reward: {avg_reward:.3f}")
        print(f"Average Episode Length: {avg_length:.1f}")
        print(f"Std Episode Length: {std_length:.1f}")

        metrics = {
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "avg_length": avg_length,
            "std_length": std_length,
            "rewards": rewards,
            "successes": successes,
            "lengths": lengths,
            "min_steps": min_steps,
        }
        self._print_legacy_summary(metrics)
        return metrics

    def visualize_testing_progress(self, n_episodes=5, explore=False):
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

    def visualize_testing_progess(self, n_episodes=5, explore=False):  # Backward compat spelling
        self.visualize_testing_progress(n_episodes=n_episodes, explore=explore)

    def _moving_average(self, data, window=5):
        if len(data) < window:
            return data
        kernel = np.ones(window) / window
        return np.convolve(data, kernel, mode="valid")

    def plot_basic_curves(self, rewards):
        if not self.delta_history:
            print("Delta history empty, skipping convergence plot.")
        else:
            plt.figure(figsize=(10, 5))
            plt.plot(self.delta_history, color="tab:blue")
            plt.xlabel("Sweep")
            plt.ylabel("Bellman residual (delta)")
            plt.title("Value Iteration: Convergence over Time")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            plt.show()

        if len(rewards) == 0:
            print("No reward data available for plotting.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(range(len(rewards)), rewards, color="tab:green", label="Episode reward")
        if len(rewards) >= 5:
            ma = self._moving_average(rewards, 5)
            plt.plot(range(4, len(rewards)), ma, label="5-ep moving avg", color="tab:orange")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Value Iteration: Reward Over Time")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_metric_trends(self, rewards, successes, lengths, min_steps):
        if len(rewards) == 0:
            print("No rollout data to visualize.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        fig.suptitle("Value Iteration Performance Metrics", fontsize=14, fontweight="bold")

        ax = axes[0][0]
        ax.plot(rewards, label="Reward", color="#1976d2")
        if len(rewards) >= 5:
            ax.plot(
                range(4, len(rewards)),
                self._moving_average(rewards, 5),
                label="5-ep MA",
                color="#42a5f5",
                linestyle="--",
            )
        ax.set_title("Learning Performance")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax_success = ax.twinx()
        success_curve = np.cumsum(successes) / (np.arange(len(successes)) + 1)
        ax_success.plot(success_curve, color="#2e7d32", label="Cumulative success")
        ax_success.set_ylabel("Success rate")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax_success.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="lower right")

        ax = axes[0][1]
        valid = min_steps > 0
        if np.any(valid):
            efficiency_ratio = np.zeros_like(lengths, dtype=float)
            efficiency_ratio[valid] = min_steps[valid] / np.maximum(lengths[valid], 1)
            ax.plot(efficiency_ratio, color="#7cb342")
            ax.set_ylim(0, 1.1)
        ax.set_title("Policy Efficiency (lower bound / actual)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Efficiency ratio")
        ax.grid(True, linestyle="--", alpha=0.4)

        ax = axes[1][0]
        if self.delta_history:
            ax.plot(self.delta_history, marker="o", color="#f57c00")
        ax.set_title("Learning Stability (delta)")
        ax.set_xlabel("Sweep")
        ax.set_ylabel("Max update")
        ax.grid(True, linestyle="--", alpha=0.4)

        ax = axes[1][1]
        total_backups = self.training_sweeps * len(self.valid_states)
        metrics = [
            ("Sweeps", self.training_sweeps),
            ("Runtime (s)", self.training_time),
            ("Backups", total_backups),
        ]
        bars = ax.bar(
            [m[0] for m in metrics],
            [m[1] for m in metrics],
            color=["#4dd0e1", "#26c6da", "#00acc1"],
            alpha=0.9,
        )
        for bar, (_, value) in zip(bars, metrics):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}" if isinstance(value, float) else f"{int(value)}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        ax.set_title("Training Efficiency")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def summarize_metrics(self, rewards, successes, lengths, min_steps):
        print("\n=== Value-Iteration Evaluation Summary ===")

        print("-- Learning Performance --")
        print(f"Average reward: {np.mean(rewards):.2f}")
        print(f"Success rate:  {np.mean(successes):.1%}")

        print("\n-- Policy Efficiency --")
        valid = min_steps > 0
        if np.any(valid):
            efficiency = lengths[valid] / np.maximum(min_steps[valid], 1)
            print(f"Avg episode length:        {np.mean(lengths):.2f} steps")
            print(f"Avg Manhattan lower bound: {np.mean(min_steps[valid]):.2f} steps")
            print(f"Length / lower-bound ratio:{np.mean(efficiency):.2f}x")
        else:
            print("Not enough data to compute efficiency ratio.")

        print("\n-- Learning Stability --")
        if self.delta_history:
            print(f"Initial delta: {self.delta_history[0]:.2e}")
            print(f"Final delta:   {self.delta_history[-1]:.2e}")
        else:
            print("Delta history empty.")

        print("\n-- Training Efficiency --")
        print(f"Sweeps executed: {self.training_sweeps}")
        print(f"Runtime: {self.training_time:.3f}s")
        total_backups = self.training_sweeps * len(self.valid_states)
        print(f"Total state backups: {total_backups}")

    def _print_legacy_summary(self, metrics):
        print("\n=== Summary ===")
        print(f"Optimal path length: {metrics['avg_length']:.1f}")
        print(f"Final Episode Reward: {metrics['avg_reward']:.3f}")
        print(f"Final Success Rate: {metrics['success_rate']:.1%}")
        print(f"Episode to convergence: {self.training_sweeps}")
        print(f"Total Computational Time: {self.training_time:.4f}s")


def run_training_and_diagnostics(
    train_episodes_env=None, eval_episodes_env=None, eval_rollouts=200, test_rollouts=100
):
    training_env = train_episodes_env or MapEnv(render_mode=None)
    agent = ValueIterationAgent(training_env)
    agent.train()
    agent.visualize_testing_progress()

    eval_env = eval_episodes_env or MapEnv(render_mode=None)
    rewards, successes, lengths, min_steps = agent.evaluate_policy(eval_env, n_episodes=eval_rollouts)
    agent.summarize_metrics(rewards, successes, lengths, min_steps)

    metrics = {
        "rewards": rewards,
        "successes": successes,
        "lengths": lengths,
        "min_steps": min_steps,
    }
    agent.plot_basic_curves(rewards)
    agent.plot_metric_trends(rewards, successes, lengths, min_steps)

    if test_rollouts:
        agent.test_agent(eval_env, n_episodes=test_rollouts)

    if eval_episodes_env is None:
        eval_env.close()
    if train_episodes_env is None:
        training_env.close()

    return agent, metrics


if __name__ == "__main__":
    training_env = MapEnv(render_mode=None)
    agent = ValueIterationAgent(training_env)
    agent.train()
    agent.visualize_testing_progress()

    evaluation_env = MapEnv(render_mode=None)
    rollouts = agent.test_agent(evaluation_env, n_episodes=200)
    agent.summarize_metrics(
        rollouts["rewards"], rollouts["successes"], rollouts["lengths"], rollouts["min_steps"]
    )
    agent.plot_basic_curves(rollouts["rewards"])
    agent.plot_metric_trends(
        rollouts["rewards"], rollouts["successes"], rollouts["lengths"], rollouts["min_steps"]
    )

    # Optional interactive visualization
    # agent.visualize_testing_progress(5)

    evaluation_env.close()
    training_env.close()
