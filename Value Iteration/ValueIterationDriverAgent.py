# Value Iteration/DriverAgent.py
import numpy as np
from matplotlib import pyplot as plt
from Map import MapEnv
from time import perf_counter

GAMMA = 0.95 # Discount factor
THRESHOLD = 1e-4 # Bellman residual threshold for convergence
EPSILON = 0.05 # Exploration rate for epsilon-greedy policy
MAX_EVAL_STEPS = 200 # Max steps per episode during evaluation

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

    def evaluate_policy(self, env, n_episodes=20000):
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
                while not done and steps < MAX_EVAL_STEPS:
                    action = self.greedy_action(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    steps += 1
                    done = terminated or truncated
                rewards.append(episode_reward)
                successes.append(float(terminated))
                lengths.append(steps)
                manhattan = np.sum(np.abs(start_driver - start_dest))
                min_steps.append(manhattan)
        finally:
            self.epsilon = old_epsilon
        return (
            np.array(rewards, dtype=float),
            np.array(successes, dtype=float),
            np.array(lengths, dtype=float),
            np.array(min_steps, dtype=float),
        )

    def _moving_average(self, data, window=5):
        if len(data) < window:
            return data
        kernel = np.ones(window) / window
        return np.convolve(data, kernel, mode="valid")

    def plot_metrics(self, rewards, successes, window=5):
        if len(self.delta_history) == 0:
            print("No convergence data to plot.")
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(self.delta_history, marker="o")
        axes[0].set_title("Bellman Residual per Sweep")
        axes[0].set_xlabel("Sweep")
        axes[0].set_ylabel("Max state update (delta)")
        axes[0].set_yscale("log")
        axes[0].grid(True, linestyle="--", alpha=0.4)

        axes[1].plot(rewards, label="Episode reward", alpha=0.6)
        if len(rewards) >= window:
            ma = self._moving_average(rewards, window)
            axes[1].plot(range(window - 1, len(rewards)), ma, label=f"{window}-ep moving avg")
        axes[1].set_title("Evaluation Reward per Episode")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Reward")
        success_rate = np.cumsum(successes) / (np.arange(len(successes)) + 1)
        ax_success = axes[1].twinx()
        ax_success.plot(success_rate, label="Cumulative success rate", color="green", linestyle="--")
        ax_success.set_ylabel("Success rate")
        ax_success.set_ylim(0, 1.05)
        lines, labels = axes[1].get_legend_handles_labels()
        lines2, labels2 = ax_success.get_legend_handles_labels()
        axes[1].legend(lines + lines2, labels + labels2, loc="lower right")
        axes[1].grid(True, linestyle="--", alpha=0.4)

        fig.tight_layout()
        plt.show()

    def plot_metric_matrix(self, rewards, successes, lengths, min_steps):
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        fig.suptitle("Value Iteration Performance Matrix", fontsize=14, fontweight="bold")

        # Learning Performance
        ax = axes[0][0]
        ax.axis("off")
        lp_text = (
            "Learning Performance\n"
            f"Avg reward: {np.mean(rewards):.2f}\n"
            f"Success rate: {np.mean(successes):.1%}"
        )
        ax.text(
            0.5,
            0.5,
            lp_text,
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="#e0f7fa"),
        )

        # Policy Efficiency
        ax = axes[0][1]
        ax.axis("off")
        valid = min_steps > 0
        if np.any(valid):
            efficiency = lengths[valid] / min_steps[valid]
            pe_text = (
                "Policy Efficiency\n"
                f"Avg length: {np.mean(lengths):.2f}\n"
                f"Lower bound: {np.mean(min_steps[valid]):.2f}\n"
                f"Ratio: {np.mean(efficiency):.2f}x"
            )
        else:
            pe_text = "Policy Efficiency\nNot enough data"
        ax.text(
            0.5,
            0.5,
            pe_text,
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="#f1f8e9"),
        )

        # Learning Stability
        ax = axes[1][0]
        ax.axis("off")
        if self.delta_history:
            stability_text = (
                "Learning Stability\n"
                f"Initial delta: {self.delta_history[0]:.2e}\n"
                f"Final delta: {self.delta_history[-1]:.2e}"
            )
        else:
            stability_text = "Learning Stability\nNo delta history"
        ax.text(
            0.5,
            0.5,
            stability_text,
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="#fff3e0"),
        )

        # Training Efficiency
        ax = axes[1][1]
        ax.axis("off")
        total_backups = self.training_sweeps * len(self.valid_states)
        te_text = (
            "Training Efficiency\n"
            f"Sweeps: {self.training_sweeps}\n"
            f"Runtime: {self.training_time:.2f}s\n"
            f"Backups: {total_backups}"
        )
        ax.text(
            0.5,
            0.5,
            te_text,
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="#fce4ec"),
        )

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
            efficiency = lengths[valid] / min_steps[valid]
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
    eval_env = MapEnv(render_mode=None)
    rewards, successes, lengths, min_steps = agent.evaluate_policy(eval_env, n_episodes=50)
    agent.summarize_metrics(rewards, successes, lengths, min_steps)
    agent.test_agent(eval_env, 20000)
    agent.plot_metrics(rewards, successes)
    agent.plot_metric_matrix(rewards, successes, lengths, min_steps)
    eval_env.close()
    agent.visualize_testing_progess(10)
    training_env.close()
