import gymnasium as gym
import numpy as np
from typing import Optional

# DEFINE THE ENVIRONMENT CLASS
class MapEnv(gym.Env):
    def __init__(self, size: int = 8, max_steps: int = 100):
        self.size = size
        self.max_steps = max_steps
        
        # Initialize driver and destination locations
        self._driver_location = np.array([-1, -1], dtype=np.int64)
        self._destination_location = np.array([-1, -1], dtype=np.int64)

        # Define observation space
        self.observation_space = gym.spaces.Dict({
            "driver": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "destination": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
        })

        # Define actions
        self.action_space = gym.spaces.Discrete(4)

        # Map action to a movement dirction
        self._action_to_direction = {
            0: np.array([1, 0]),   # Move right (positive x)
            1: np.array([0, 1]),   # Move up (positive y)
            2: np.array([-1, 0]),  # Move left (negative x)
            3: np.array([0, -1]),  # Move down (negative y)
        }

    # CONSTRUCTING OBSERVATION AND INFO
    def _get_obs(self):
        return {"driver": self._driver_location, "destination": self._destination_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._driver_location - self._destination_location, ord=1
            )
        }

    # RESET FUNCTION
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Randomly place driver location on Map
        self._driver_location = self.np_random.integers(0, self.size, size=2, dtype=np.int64)

        # Randomly place destination location on Map but not at the same position as driver
        self._destination_location = self._driver_location
        # Keep looping until a different location is found
        while np.array_equal(self._destination_location, self._driver_location):
            self._destination_location = self.np_random.integers(0, self.size, size=2, dtype=np.int64)

        # Reset step counter for truncation handling
        self.current_step = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    # STEP FUNCTION
    def step(self, action: int):
        # Map action to direction
        direction = self._action_to_direction[action]

        # Update driver location with boundary checks
        self._driver_location = np.clip(
            self._driver_location + direction, 0, self.size - 1
        )

        # Check if agent reached the destination
        terminated = np.array_equal(self._driver_location, self._destination_location)

        # Track steps and truncate if exceeding max_steps
        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        # Reward structure
        distance = np.linalg.norm(self._driver_location - self._destination_location)
        reward = 100 if terminated else -0.01 * distance

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
    
# TEST ENVIRONMENT
if __name__ == "__main__":
    env = MapEnv(size=8)
    obs, info = env.reset(seed=42)
    print("Initial observation:", obs, "info:", info)

    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step}: Action={action}, Obs={obs}, Reward={reward}, Done={terminated}")
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
    