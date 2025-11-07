import gymnasium as gym
import numpy as np
from typing import Optional
import pygame
from gymnasium.envs.registration import register


class MapEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size: int = 8, max_steps: int = 100, render_mode="human"):
        self.size = size
        self.max_steps = max_steps
        self.window_size = 512

        # Initialize driver and destination locations
        self._driver_location = np.array([-1, -1], dtype=np.int64)
        self._destination_location = np.array([-1, -1], dtype=np.int64)

        # Initialize fixed obstacle
        self._obstacles = [
            np.array([3, 3]), 
            np.array([3, 4]), 
            np.array([4, 3]), 
            np.array([4, 4]),
            np.array([1, 6]),
            np.array([2, 6]),
            np.array([6, 1]),
            np.array([6, 2]),
        ]

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.current_step = 0

        # Define observation space
        self.observation_space = gym.spaces.Dict({
            "driver": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "destination": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
        })

        # Define actions
        self.action_space = gym.spaces.Discrete(4)

        # Map action to a movement direction
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

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Randomly place driver location on Map
        self._driver_location = self.np_random.integers(0, self.size, size=2, dtype=np.int64)


        while self._is_obstacle(self._driver_location):
            self._driver_location = self.np_random.integers(0, self.size, size=2, dtype=np.int64)

        # Randomly place destination location on Map but not at the same position as driver
        self._destination_location = self._driver_location
        # Keep looping until a different location is found
        while np.array_equal(self._destination_location, self._driver_location) or self._is_obstacle(self._destination_location):
            self._destination_location = self.np_random.integers(0, self.size, size=2, dtype=np.int64)

        # Reset step counter for truncation handling
        self.current_step = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def _is_obstacle(self, position):
        for obstacle in self._obstacles:
            if np.array_equal(position, obstacle):
                return True
        return False

    def step(self, action: int):
        # Map action to direction
        direction = self._action_to_direction[action]

        # Update driver location with boundary checks
        new_location = np.clip(
            self._driver_location + direction, 0, self.size - 1
        )

        hit_obstacle = self._is_obstacle(new_location)
        if not hit_obstacle: 
            self._driver_location = new_location

        # Check if agent reached the destination
        terminated = np.array_equal(self._driver_location, self._destination_location)

        self.current_step += 1

        # Calculate reward
        if terminated:
            # Success!
            reward = 100
            truncated = False
        elif self._is_obstacle(self._driver_location):
            # Somehow ended up on obstacle
            reward = -100
            truncated = True
        else:
            # Normal step: small penalty based on distance
            distance = np.linalg.norm(self._driver_location - self._destination_location)
            reward = -0.01 * distance
            
            # Extra penalty for trying to move into obstacle
            if hit_obstacle:
                reward -= 10
            
            # Check if max steps exceeded
            truncated = self.current_step >= self.max_steps

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the destination
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._destination_location,
                (pix_square_size, pix_square_size),
            ),
        )

        # Next, we draw the obstacles
        for obstacle in self._obstacles:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    pix_square_size * obstacle,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._driver_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

# TEST ENVIRONMENT
if __name__ == "__main__":
    env = MapEnv()
    obs, info = env.reset(seed=42)
    print("Initial observation:", obs, "info:", info)

    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step}: Action={action}, Obs={obs}, Reward={reward}, Done={terminated}")
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
    
    register(
        id="gymnasium_env/TaxiDriverMap-v0",
        entry_point="gymnasium.env.envs:TaxiDriverMapEnv",
    )
