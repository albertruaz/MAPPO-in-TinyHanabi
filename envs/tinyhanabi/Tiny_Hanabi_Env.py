import numpy as np
from gym.spaces import Discrete

class Environment(object):
    def seed(self, seed):
        raise NotImplementedError

    def reset(self, config=None):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

class TinyHanabiEnv(Environment):
    """
    A matrix-game-like environment inspired by tiny-hanabi rules:
    - 2 players
    - Each player has one of two possible observations: {0,1}
    - Each player chooses one of three actions: {0,1,2}
    - The payoff is given by a predefined payoff_values matrix:
      shape: (2, 2, 3, 3)
      indexing: payoff_values[obs_p0, obs_p1, action_p0, action_p1]
    - The episode ends after one step.
    """

    def __init__(self, args, seed):
        self.num_players = args.num_agents
        assert self.num_players == 2, "This example supports only 2 players"
        self._seed = seed
        if not hasattr(args, 'use_obs_instead_of_state'):
            args.use_obs_instead_of_state = False
        
        self.obs_instead_of_state = args.use_obs_instead_of_state

        # Define the action space (3 actions: 0,1,2)
        self.action_space = [Discrete(3) for _ in range(self.num_players)]

        # Observation dimension:
        # 여기서는 각 player의 관측을 단순화해서:
        # obs는 단순히 [obs_value(one-hot 2차원), current_player_one_hot(2)] 정도로 가정하자.
        # obs_value ∈ {0,1}, one-hot로 표현하면 2차원
        # current_player_one_hot(2차원)
        # 총 obs_dim = 2(관측 one-hot) + 2(플레이어 one-hot) = 4
        self.obs_dim = 4
        self.observation_space = [[self.obs_dim] for _ in range(self.num_players)]

        # share_obs: 두 플레이어의 관측을 모두 합친다고 가정
        # share_obs_dim = obs_dim * 2 = 8
        self.share_obs_dim = self.obs_dim * self.num_players
        self.share_observation_space = [[self.share_obs_dim] for _ in range(self.num_players)]

        self.seed(seed)

        # payoff_values 정의 (원문 코드에 있는 예시를 그대로 사용)
        self.payoff_values = np.array([
          [[[10, 0, 0], [4, 8, 4], [10, 0, 0]],
           [[0, 0, 10], [4, 8, 4], [0, 0, 10]]],
          [[[0, 0, 10], [4, 8, 4], [0, 0, 0]],
           [[10, 0, 0], [4, 8, 4], [10, 0, 0]]]
        ])

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def reset(self, choose=True):
        if not choose:
            # Return zero obs
            obs = np.zeros(self.obs_dim)
            share_obs = np.zeros(self.share_obs_dim)
            # 3 actions: always available
            available_actions = np.ones(3)
            return obs, share_obs, available_actions

        # 각 플레이어의 관측을 랜덤하게 선택(0 또는 1)
        self.obs_for_players = [np.random.randint(0,2) for _ in range(self.num_players)]
        self.current_player = 0  # 첫 턴에 player 0이 행동한다고 가정

        obs, share_obs, avail = self._get_obs()
        return obs, share_obs, avail

    def close(self):
        pass

    def _get_obs(self):
        # 현재 플레이어의 관측:
        # obs_value = self.obs_for_players[self.current_player]
        # one-hot encode obs_value
        obs_val = np.zeros(2)
        obs_val[self.obs_for_players[self.current_player]] = 1.0

        # current_player_one_hot
        player_one_hot = np.zeros(self.num_players)
        player_one_hot[self.current_player] = 1.0

        obs = np.concatenate([obs_val, player_one_hot], axis=0)

        # share_obs: 모든 플레이어 관측을 합침
        share_obs_all = []
        for pid in range(self.num_players):
            c_oh = np.zeros(2)
            c_oh[self.obs_for_players[pid]] = 1.0
            p_oh = np.zeros(self.num_players)
            p_oh[pid] = 1.0
            p_obs = np.concatenate([c_oh, p_oh], axis=0)
            share_obs_all.append(p_obs)

        share_obs = np.concatenate(share_obs_all, axis=0)
        # Available actions: 3개 모두 가능
        available_actions = np.ones(3)
        return obs, share_obs, available_actions

    def num_moves(self):
        # Three possible moves: 0, 1, 2
        return 3

    def step(self, action):
        # action이 어떤 형태로 들어와도 (num_players,1) 형태로 변환
        action = np.array(action)
        
        # action이 int 하나로 들어온 경우 예: action=0
        # 혹은 action이 (num_players,) 형태로 들어올 수도 있음
        # onpolicy에서 일반적으로 (num_envs, num_agents, 1) 형태를 쓰지만,
        # single-env일 경우 (num_agents, 1) 형태여야 함.
        if action.ndim == 0:
            # single int
            action = np.array([[action]] * self.num_players)
        elif action.ndim == 1:
            # shape: (num_players,) 일 경우 reshape
            action = action.reshape(-1, 1)
        # else: action.ndim == 2 일 경우 이미 (num_players,1) 일 가능성 높음


        # step 함수 내부에서:
        # action은 (num_envs, num_agents, 1) 형태로 들어온다고 가정.
        # num_envs=1인 경우 (1, num_agents, 1) 형태일 것이므로 squeeze 해줍니다.
        if action.ndim == 3:
            # action.shape = (1, num_agents, 1)
            # 첫 번째 축(num_envs)은 1이므로 squeeze
            action = np.squeeze(action, axis=0)  # 이제 (num_agents, 1) 형태가 됩니다.

        # 이제 action.shape == (num_agents, 1)을 만족해야 함
        assert action.shape == (self.num_players, 1), f"Expected (2,1), got {action.shape}"

        a0 = action[0][0]
        a1 = action[1][0]

        rew = self.payoff_values[self.obs_for_players[0], self.obs_for_players[1], a0, a1]
        done = True
        rewards = [[rew]] * self.num_players
        infos = {'score': rew}

        obs = np.zeros(self.obs_dim)
        share_obs = np.zeros(self.share_obs_dim)
        available_actions = np.ones(3)

        return obs, share_obs, rewards, done, infos, available_actions
