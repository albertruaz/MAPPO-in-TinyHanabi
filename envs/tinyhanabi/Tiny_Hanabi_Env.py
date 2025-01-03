import numpy as np
from gym.spaces import Discrete, Box
import gym

class TinyHanabiEnv(gym.Env):
    """
    A simple turn-based environment with two players.
    Each episode ends after both players have taken one action each.
    """

    def __init__(self, all_args, seed=None):
        """
        Args:
            all_args: Config object containing training arguments.
            seed: Optional random seed for reproducibility.
        """
        super(TinyHanabiEnv, self).__init__()

        self.all_args = all_args
        self.num_players = 2

        # Define action space: each player can choose among 3 discrete actions.
        # Here, we store the action_space for each player in a list,
        # but for a single-gym environment interface, let's just store the first player's space.
        # (We will handle player switching manually.)
        self.action_space = Discrete(3)

        # Define observation space:
        # Player 0 sees shape (2,) -> one-hot vector representing an internal state 0 or 1
        # Player 1 sees shape (5,) -> (2,) one-hot for own state plus (3,) one-hot for player0's action
        # For the gym.Env single observation_space, let's just set the maximum dimension needed
        # and handle the real shape at runtime (player 0 vs player 1).
        # The largest observation shape is (5,) for Player 1, so we use a Box with shape (5,).
        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32
        )

        # Internal states
        self.obs_for_players = [0, 0]  # each can be 0 or 1
        self.current_player = 0        # 0 or 1
        self.last_actions = [None, None]

        # payoff_values for reward calculation (given in the example)
        self.payoff_values = np.array([
            [
                [[10, 0, 0], [4, 8, 4], [10, 0, 0]],
                [[0, 0, 10], [4, 8, 4], [0, 0, 10]]
            ],
            [
                [[0, 0, 10], [4, 8, 4], [0, 0, 0]],
                [[10, 0, 0], [4, 8, 4], [10, 0, 0]]
            ]
        ])

        self.seed(seed)

    def seed(self, seed=None):
        """
        Sets the random seed for NumPy. If no seed is given, defaults to 0.
        """
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(0)

    def reset(self):
        """
        Resets the environment by randomly choosing each player's internal state (0 or 1).
        Returns observation for the current player (player 0).
        """
        self.obs_for_players = [np.random.randint(0, 2) for _ in range(self.num_players)]
        self.current_player = 0
        self.last_actions = [None, None]

        return self._get_obs(self.current_player)

    def _get_obs(self, player_id):
        """
        Returns the observation for the requested player_id.
        Player 0: shape (2,). One-hot.
        Player 1: shape (5,). (2,) one-hot + (3,) one-hot of player0's action.
        """
        if player_id == 0:
            # (2,) one-hot
            obs_val = np.zeros(2, dtype=np.float32)
            obs_val[self.obs_for_players[0]] = 1.0

            # We'll pad to shape (5,) to match environment's observation_space
            padded_obs = np.zeros(5, dtype=np.float32)
            padded_obs[:2] = obs_val
            return padded_obs
        else:
            # Player 1 -> shape (5,)
            obs_val = np.zeros(2, dtype=np.float32)
            obs_val[self.obs_for_players[1]] = 1.0
            a0_oh = np.zeros(3, dtype=np.float32)
            if self.last_actions[0] is not None:
                a0_oh[self.last_actions[0]] = 1.0

            return np.concatenate([obs_val, a0_oh])  # (2,) + (3,) = (5,)

    def step(self, action):
        """
        Perform a step in the environment for the current player.
        Returns: (observation, reward, done, info)
        """
        # Store the action
        self.last_actions[self.current_player] = action

        # If we just acted as player 0, switch to player 1
        if self.current_player == 0:
            self.current_player = 1
            # Reward is 0 for now because we haven't ended the episode
            # done is False because player 1 hasn't acted yet
            return self._get_obs(self.current_player), 0.0, False, {}

        else:
            # Player 1 just acted, so the episode ends. Let's compute the final reward.
            a0 = self.last_actions[0]
            a1 = self.last_actions[1]
            rew = self.payoff_values[
                self.obs_for_players[0],
                self.obs_for_players[1],
                a0,
                a1
            ]
            # Once player1 acts, the game ends
            done = True
            info = {"episode_score": rew}

            # We won't bother with next obs because done=True
            return np.zeros(5, dtype=np.float32), rew, done, info

    def close(self):
        """
        Close environment if needed. (No-op here.)
        """
        pass
