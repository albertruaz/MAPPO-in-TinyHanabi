import os
import numpy as np
import torch
import time
from gym.spaces import Discrete, Box
import wandb

from onpolicy.utils.shared_buffer import SharedReplayBuffer
from onpolicy.utils.valuenorm import ValueNorm

def _t2n(x):
    """Convert torch.Tensor to numpy."""
    return x.detach().cpu().numpy()

class TinyHanabiRunner:
    """
    A minimal runner for TinyHanabi that does NOT inherit from the base runner.
    It connects the existing TinyHanabiEnv, rMAPPO (or MAPPO), and forward functions.
    """

    def __init__(self, config):
        """
        Args:
            config (dict): {
                "all_args": all_args,   # argparse로부터 파싱된 하이퍼파라미터
                "envs": envs,           # TinyHanabiEnv or VecEnv
                "eval_envs": eval_envs, # (Optional) 평가 환경
                "num_agents": int,      # (예: 2)
                "device": torch.device, # CPU/GPU
                "run_dir": pathlib.Path or str # 로그/모델 저장 디렉토리
                ...
            }
        """
        self.all_args = config["all_args"]
        self.envs = config["envs"]
        self.eval_envs = config["eval_envs"]
        self.num_agents = config["num_agents"]
        self.device = config["device"]
        self.run_dir = config["run_dir"]
        self.use_wandb = self.all_args.use_wandb
        self.total_reward = 0
        self.value_normalizer = ValueNorm(input_shape=(1,), device=self.device)

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.log_dir = str(wandb.run.dir)
        else:
            self.save_dir = os.path.join(str(self.run_dir), "models")
            self.log_dir = os.path.join(str(self.run_dir), "logs")
            os.makedirs(self.save_dir, exist_ok=True)
            os.makedirs(self.log_dir, exist_ok=True)

        # Import the algorithm and policy based on user arguments
        if self.all_args.algorithm_name == "rmappo":
            from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
            self.all_args.use_recurrent_policy = True
        elif self.all_args.algorithm_name == "mappo":
            from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
            self.all_args.use_recurrent_policy = False
        else:
            raise NotImplementedError("Only rmappo or mappo are supported in this example.")

        # Extract observation/action spaces from env
        # If env is a VecEnv with [obs_space], [action_space] for each agent, handle that
        if hasattr(self.envs, "observation_space") and isinstance(self.envs.observation_space, list):
            obs_space = self.envs.observation_space[0]
            act_space = self.envs.action_space[0]
        else:
            obs_space = self.envs.observation_space
            act_space = self.envs.action_space

        # share_obs: if using centralized V, we might pass a bigger share_obs
        # For tiny hanabi, let's just reuse obs_space
        # share_observation_space = obs_space
        # 3 + 2 + 3 + 2 = 10차원으로 설정
        share_obs_space = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)


        # Instantiate policy and trainer
        self.policy = Policy(self.all_args, obs_space, share_obs_space, act_space, device=self.device)
        self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)
        self.rnn_states_actor = np.zeros(
            (1, 1, self.all_args.recurrent_N, self.all_args.hidden_size)
        )
        self.rnn_states_critic = np.zeros(
            (1, 1, self.all_args.recurrent_N, self.all_args.hidden_size)
        )
        self.action_log_probs = None
        self.value_preds = None

        self.buffer = SharedReplayBuffer(
            self.all_args,
            self.num_agents,
            obs_space,
            share_obs_space,
            act_space
        )
        
        # Add reward history for recent 50 episodes
        self.recent_rewards = []
        self.recent_window = 50
    
    def run(self):
        episodes = int(self.all_args.num_env_steps // self.all_args.n_rollout_threads)
        for batch_i in range(episodes):
            batch_reward = self.run_episode(batch_i)
            if (batch_i % self.all_args.save_interval == 0) or (batch_i == episodes - 1):
                self.save()

    def run_episode(self,batch_i):
        # print("This is run_episode: ", ep_i)
        for ep_i in range(self.all_args.n_rollout_threads):
            obs, only_obs = self.envs.reset() # (5,)
            done = False
            masks = torch.ones((1, self.num_agents, 1), device=self.device)
            
            # 1) agent 1
            action = self.select_action(obs)
            new_obs, reward, done, info = self.envs.step(action)
            share_obs = self.envs.get_full_obs()
            self.buffer.insert(
                share_obs=share_obs,
                obs=obs,
                rnn_states_actor=self.rnn_states_actor,
                rnn_states_critic=self.rnn_states_critic,
                action_log_probs= self.action_log_probs,
                value_preds= self.value_preds,
                actions=action,
                rewards=np.array([0]),
                masks=np.array([[[1]]], dtype=np.float32)
            )

            # 2) agent 2
            obs = new_obs
            action = self.select_action(obs)
            new_obs, reward, done, info = self.envs.step(action)
            share_obs = self.envs.get_full_obs()
            self.buffer.insert(
                share_obs=share_obs,
                obs=obs,
                rnn_states_actor=self.rnn_states_actor,
                rnn_states_critic=self.rnn_states_critic,
                action_log_probs= self.action_log_probs,
                value_preds= self.value_preds,
                actions=action,
                rewards=np.array([reward]),
                masks=np.array([[[0]]], dtype=np.float32)
            )
            

            # 3) Save Data
            self.total_reward += reward
            if self.use_wandb:
                # Calculate recent 50 average reward
                self.recent_rewards.append(reward)
                if len(self.recent_rewards) > self.recent_window:
                    self.recent_rewards.pop(0)  # Remove oldest reward
                recent_avg = np.mean(self.recent_rewards)

                wandb.log({
                    "average_reward": self.total_reward/(batch_i*self.all_args.n_rollout_threads + ep_i + 1),
                    "recent_50_average_reward": recent_avg,
                    f"(detail) reward when obs is {only_obs}": reward
                    # "episode_reward": reward
                })
            
        self.buffer.compute_returns(next_value=np.array([0]), value_normalizer=self.value_normalizer)
        train_infos = self.train()
        self.buffer.after_update()
        # wandb logging
        
            

        return reward

    @torch.no_grad()
    def select_action(self, obs):
        self.trainer.prep_rollout()
        
        cent_obs = self.envs.get_full_obs()

        masks = torch.ones((1, 1, 1), device=self.device)
        obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0).unsqueeze(0) #torch.Size([1, 1, 10])
        cent_obs_tensor = torch.FloatTensor(cent_obs).to(self.device).unsqueeze(0).unsqueeze(0) #torch.Size([1, 1, 10])

        
        values, actions, action_log_probs, new_rnn_states_actor, new_rnn_states_critic = self.policy.get_actions(
            cent_obs=cent_obs_tensor,
            obs=obs_tensor,
            rnn_states_actor=self.rnn_states_actor,
            rnn_states_critic=self.rnn_states_critic,
            masks=masks,
            available_actions=None,
            deterministic=False
        )

        self.action_log_probs = _t2n(action_log_probs)
        self.value_preds = _t2n(values)

        action_np = _t2n(actions).squeeze()
        return action_np

    def train(self):
        """
        Minimal train placeholder: calls trainer.prep_training() then does a dummy update.
        (Real usage: gather rollouts in a buffer, call trainer.train(buffer), etc.)
        """
        self.trainer.prep_training()
        train_info = self.trainer.train(self.buffer)
        self.buffer.after_update()
        return train_info

    def save(self):
        """
        Save policy's actor and critic.
        """
        actor_state = self.trainer.policy.actor.state_dict()
        critic_state = self.trainer.policy.critic.state_dict()
        torch.save(actor_state, os.path.join(self.save_dir, "actor.pt"))
        torch.save(critic_state, os.path.join(self.save_dir, "critic.pt"))

    def restore(self, model_dir):
        """
        Load policy's actor and critic.
        """
        actor_state = torch.load(os.path.join(model_dir, "actor.pt"), map_location=self.device)
        critic_state = torch.load(os.path.join(model_dir, "critic.pt"), map_location=self.device)
        self.trainer.policy.actor.load_state_dict(actor_state)
        self.trainer.policy.critic.load_state_dict(critic_state)

    @torch.no_grad()
    def eval_policy(self, eval_episodes=100):
        """
        Evaluate the trained policy over a set number of episodes.
        """
        eval_envs = self.eval_envs
        eval_scores = []
        
        for episode in range(eval_episodes):
            obs, only_obs = eval_envs.reset()
            done = False
            total_reward = 0

            masks = torch.ones((1, self.num_agents, 1), device=self.device)

            while not done:
                # action = self.select_action(obs, self.rnn_states_actor, self.rnn_states_critic, masks)
                action = self.select_action(obs)
                obs, reward, done, info = eval_envs.step(action)
                total_reward += reward

            eval_scores.append(total_reward)
            # print(f"Episode {episode + 1}: Total Reward: {total_reward}")

            self.total_reward += total_reward
            wandb.log({
                "average_reward": self.total_reward/(episode + 1),
                "reward": total_reward,
                f"(detail) reward when obs is {only_obs}": total_reward
                # "episode_reward": reward
            })

        avg_score = np.mean(eval_scores)
        print(f"\nAverage Evaluation Reward over {eval_episodes} episodes: {avg_score}")