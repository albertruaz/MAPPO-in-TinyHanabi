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
    턴 기반 예시 환경:
    - 플레이어가 2명 (player 0, player 1)
    - 한 에피소드가 총 2스텝:
      1) player 0(첫 번째 에이전트)이 obs0만 보고 a0 결정
      2) player 1(두 번째 에이전트)이 obs1 + a0를 보고 a1 결정
         그리고 payoff_values로부터 보상 계산 후 에피소드 끝
    """
    def __init__(self, seed=None):
        self.num_players = 2
        self.action_space = [Discrete(3) for _ in range(self.num_players)]
        self.obs_for_players = [0, 0]  # reset에서 갱신
        self.current_player = 0
        self.last_actions = [None, None]
        self.seed(seed)
        # payoff_values - shape (2,2,3,3)
        self.payoff_values = np.array([
            [[[10, 0, 0], [4, 8, 4], [10, 0, 0]],
             [[0, 0, 10], [4, 8, 4], [0, 0, 10]]],
            [[[0, 0, 10], [4, 8, 4], [0, 0, 0]],
             [[10, 0, 0], [4, 8, 4], [10, 0, 0]]]
        ])

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(0)

    def reset(self):
        # 0 또는 1로 플레이어별 관측을 랜덤 선택
        self.obs_for_players = [np.random.randint(0,2) for _ in range(self.num_players)]
        self.current_player = 0
        self.last_actions = [None, None]
        # 첫 에이전트(player 0)의 관측만 반환
        return self._get_obs(player_id=0)

    def _get_obs(self, player_id): # player_id가 볼 관측을 구성해서 반환
    
        if player_id == 0:
            # 첫 번째 에이전트: 자기 관측(obs0)만 본다
            # 예: one-hot으로 만들기
            obs_val = np.zeros(2)
            obs_val[self.obs_for_players[0]] = 1
            return obs_val  # shape=(2,)

        else:
            # 두 번째 에이전트: 자기 관측(obs1) + 첫 번째 에이전트의 액션(a0)을 본다
            obs_val = np.zeros(2)
            obs_val[self.obs_for_players[1]] = 1

            # 첫 번째 에이전트의 액션 a0을 one-hot으로
            a0_oh = np.zeros(3)
            if self.last_actions[0] is not None:
                a0_oh[self.last_actions[0]] = 1

            # concat해서 shape = (2 + 3) = (5,)
            return np.concatenate([obs_val, a0_oh])

    def step(self, action):
        """
        step(action) 구조:
        1) player 0가 호출 -> a0을 저장 -> 이제 player 1의 obs를 리턴 (done=False)
        2) player 1가 호출 -> a1을 저장 -> payoff_values로 보상 계산 -> done=True
        """
        # 현재 플레이어의 액션을 기록
        self.last_actions[self.current_player] = action

        if self.current_player == 0:
            # 이제 player 0은 액션을 냈으니 player 1 차례로 전환
            self.current_player = 1
            # player 1이 볼 관측을 반환
            obs = self._get_obs(player_id=1)
            reward = 0.0  # 아직은 보상 없음
            done = False
            info = {}
            return obs, reward, done, info

        else:
            # 이제 player 1의 액션까지 나왔으니 payoff 계산
            a0 = self.last_actions[0]
            a1 = self.last_actions[1]

            rew = self.payoff_values[
                self.obs_for_players[0],
                self.obs_for_players[1],
                a0,
                a1
            ]
            done = True
            info = {"episode_score": rew}
            # 에피소드 끝났으므로 observation은 None
            return None, rew, done, info

    def close(self):
        pass
