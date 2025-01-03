import time
import wandb
import os
import numpy as np
from itertools import chain
import torch

from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.shared.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class TinyHanabiRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for TinyHanabi."""
    def __init__(self, config):
        super(TinyHanabiRunner, self).__init__(config)
        self.true_total_num_steps = 0
        self.episode_length = 0

    def run(self):
        print("~~~~~~~~~~~~~~~~1run start")
        # turn_xxx는 한 에피소드 내에서 에이전트 관측/액션 등을 저장하기 위한 변수
        self.turn_obs = np.zeros((self.n_rollout_threads, self.num_agents, *self.buffer.obs.shape[3:]), dtype=np.float32)
        self.turn_share_obs = np.zeros((self.n_rollout_threads, self.num_agents, *self.buffer.share_obs.shape[3:]), dtype=np.float32)
        self.turn_available_actions = np.zeros((self.n_rollout_threads, self.num_agents, *self.buffer.available_actions.shape[3:]), dtype=np.float32)
        self.turn_values = np.zeros((self.n_rollout_threads, self.num_agents, *self.buffer.value_preds.shape[3:]), dtype=np.float32)
        self.turn_actions = np.zeros((self.n_rollout_threads, self.num_agents, *self.buffer.actions.shape[3:]), dtype=np.float32)
        self.turn_action_log_probs = np.zeros((self.n_rollout_threads, self.num_agents, *self.buffer.action_log_probs.shape[3:]), dtype=np.float32)
        self.turn_rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, *self.buffer.rnn_states.shape[3:]), dtype=np.float32)
        self.turn_rnn_states_critic = np.zeros_like(self.turn_rnn_states)
        self.turn_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.turn_active_masks = np.ones_like(self.turn_masks)
        self.turn_bad_masks = np.ones_like(self.turn_masks)
        self.turn_rewards = np.zeros((self.n_rollout_threads, self.num_agents, *self.buffer.rewards.shape[3:]), dtype=np.float32)

        ####### 기존의 warmup-code
        # reset env
        self.reset_choose = np.ones(self.n_rollout_threads) == 1.0
        obs, share_obs, available_actions = self.envs.reset(self.reset_choose)
        share_obs = share_obs if self.use_centralized_V else obs
        # replay buffer 초기 상태 설정
        self.use_obs = obs.copy()
        self.use_share_obs = share_obs.copy()
        self.use_available_actions = available_actions.copy()
        ######

        start = time.time()
        # 기존 episode_length는 tinyhanabi에서 사실상 1
        episodes = int(self.num_env_steps) // self.n_rollout_threads

        for episode in range(episodes):
            print("~~~~~~~~~~~~~~~~2run episode: ", episode)
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            self.scores = []
            self.episode_length = 0

            # TinyHanabi에서 episode_length를 제약을 두지 않고 실행
            for step in range(10): # 사실상 MAX
                self.episode_length += 1 
                self.reset_choose = np.zeros(self.n_rollout_threads) == 1.0
                self.collect()  # 모든 에이전트 동시 액션 수집 및 환경 스텝

                # buffer의 마지막 인덱스 처리 (TinyHanabi는 단일 step이므로 step=0일 때 바로 처리)
                if episode > 0:
                    # 이전 에피소드 마지막 자료 처리
                    self.compute()
                    train_infos = self.train()

                # 이번 턴(한 step) 데이터를 버퍼에 삽입
                self.buffer.chooseinsert(self.turn_share_obs,
                                         self.turn_obs,
                                         self.turn_rnn_states,
                                         self.turn_rnn_states_critic,
                                         self.turn_actions,
                                         self.turn_action_log_probs,
                                         self.turn_values,
                                         self.turn_rewards,
                                         self.turn_masks,
                                         self.turn_bad_masks,
                                         self.turn_active_masks,
                                         self.turn_available_actions)

                # 에피소드 종료 후 환경 reset
                obs, share_obs, available_actions = self.envs.reset(np.ones(self.n_rollout_threads, dtype=bool))
                share_obs = share_obs if self.use_centralized_V else obs
                self.use_obs = obs.copy()
                self.use_share_obs = share_obs.copy()
                self.use_available_actions = available_actions.copy()

            # 한 에피소드 종료 후 처리
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            if episode % self.log_interval == 0 and episode > 0:
                end = time.time()
                print("\n Env {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.hanabi_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                average_score = np.mean(self.scores) if len(self.scores) > 0 else 0.0
                print("average score is {}.".format(average_score))
                if self.use_wandb:
                    wandb.log({'average_score': average_score}, step=self.true_total_num_steps)
                else:
                    self.writter.add_scalars('average_score', {'average_score': average_score}, self.true_total_num_steps)

                train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
                self.log_train(train_infos, self.true_total_num_steps)

            # if episode % self.eval_interval == 0 and self.use_eval:
            #     self.eval(self.true_total_num_steps)        

    @torch.no_grad()
    def collect(self):
        
        # PPO Action 선택
        self.trainer.prep_rollout() # eval mode로 변환
        value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer.policy.get_actions(
            self.use_share_obs,
            self.use_obs,
            self.turn_rnn_states,
            self.turn_rnn_states_critic,
            self.turn_masks,
            self.use_available_actions
        )

        # PPO Action 옮기기
        self.turn_obs = self.use_obs.copy()
        self.turn_share_obs = self.use_share_obs.copy()
        self.turn_available_actions = self.use_available_actions.copy()
        self.turn_values = _t2n(value)
        self.turn_actions = _t2n(action)
        self.turn_action_log_probs = _t2n(action_log_prob)
        self.turn_rnn_states = _t2n(rnn_state)
        self.turn_rnn_states_critic = _t2n(rnn_state_critic)

        # Hanabi Env의 Action에 따른 결과
        print("!!!!!!!!!!!!!!!!Personal Checker!!!!!!!!!!!!!!!!!!!!!!")
        print(action)
        print("!!!!!!!!!!!!!!!!Check Finish!!!!!!!!!!!!!!!!!!!!!!!!!!")
        obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(_t2n(action))
        self.true_total_num_steps += self.n_rollout_threads
        share_obs = share_obs if self.use_centralized_V else obs
        self.turn_rewards = rewards.copy()

        # done 처리: TinyHanabi는 한 스텝 후 done=True
        self.turn_masks = 1 - dones
        self.turn_active_masks = 1 - dones

        self.use_obs = obs.copy()
        self.use_share_obs = share_obs.copy()
        self.use_available_actions = available_actions.copy()

        for info in infos:
            if 'score' in info.keys():
                self.scores.append(info['score'])

    def train(self):
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        self.buffer.chooseafter_update()
        return train_infos

    # @torch.no_grad()
    # def eval(self, total_num_steps):
    #     eval_envs = self.eval_envs
    #     eval_scores = []

    #     eval_reset_choose = np.ones(self.n_eval_rollout_threads) == 1.0
    #     eval_obs, eval_share_obs, eval_available_actions = eval_envs.reset(eval_reset_choose)

    #     eval_share_obs = eval_share_obs if self.use_centralized_V else eval_obs
    #     eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, *self.buffer.rnn_states.shape[3:]), dtype=np.float32)
    #     eval_rnn_states_critic = np.zeros_like(eval_rnn_states)
    #     eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

    #     # TinyHanabi 평가도 1 스텝 후 done
    #     self.trainer.prep_rollout()
    #     value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer.policy.get_actions(
    #         eval_share_obs,
    #         eval_obs,
    #         eval_rnn_states,
    #         eval_rnn_states_critic,
    #         eval_masks,
    #         eval_available_actions,
    #         deterministic=True
    #     )

    #     eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = eval_envs.step(_t2n(action))

    #     for info in eval_infos:
    #         if 'score' in info.keys():
    #             eval_scores.append(info['score'])

    #     eval_average_score = np.mean(eval_scores) if len(eval_scores) > 0 else 0.0
    #     print("eval average score is {}.".format(eval_average_score))
    #     if self.use_wandb:
    #         wandb.log({'eval_average_score': eval_average_score}, step=total_num_steps)
    #     else:
    #         self.writter.add_scalars('eval_average_score', {'eval_average_score': eval_average_score}, total_num_steps)
