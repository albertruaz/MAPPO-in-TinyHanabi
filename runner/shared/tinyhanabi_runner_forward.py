import os
import numpy as np
import torch
import time
import wandb

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

        # Prepare logging directories
        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.log_dir = str(wandb.run.dir)
        else:
            # Local directories
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
        share_observation_space = obs_space

        # Instantiate policy and trainer
        self.policy = Policy(self.all_args, obs_space, share_observation_space, act_space, device=self.device)
        self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)

        # Tracking
        self.total_env_steps = 0
        self.episode_count = 0

    def run(self):
        """
        Main training loop:
        - For a 2-step TinyHanabi environment, each episode is short.
        - We run enough episodes to cover 'num_env_steps'.
        """
        # Each episode of TinyHanabi is 2 steps, so approximate the needed episodes
        episodes = int(self.all_args.num_env_steps // 2)

        for ep_i in range(episodes):
            ep_reward = self.run_episode()
            self.episode_count += 1

            # Logging
            if ep_i % self.all_args.log_interval == 0:
                print(f"[Episode {ep_i}] Reward: {ep_reward:.2f}")

            # Save model
            if (ep_i % self.all_args.save_interval == 0) or (ep_i == episodes - 1):
                self.save()

        # Final save
        self.save()

    def run_episode(self):
        """
        Roll out one episode in the environment.
        TinyHanabi ends after player0 + player1 actions (2 steps).
        """
        obs = self.envs.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = self.select_action(obs)
            next_obs, reward, done, info = self.envs.step(action)
            self.total_env_steps += 1
            ep_reward += reward

            # (Optional) If you have a replay buffer, store transitions here
            obs = next_obs

        # (Optional) train after each episode
        train_infos = self.train()

        # wandb logging
        if self.use_wandb:
            wandb.log({
                "episode_reward": ep_reward,
                "total_env_steps": self.total_env_steps
            })

        return ep_reward

    @torch.no_grad()
    def select_action(self, obs):
        """
        Use the policy to select an action given obs.
        """
        # Switch policy/trainer to rollout mode (no gradient updates)
        self.trainer.prep_rollout()

        # Convert obs to torch tensor
        # shape: (5,) in TinyHanabi -> expand to (batch=1, agent=1, obs_dim)
        obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0).unsqueeze(0)

        # Prepare RNN states
        rnn_states_actor = torch.zeros(
            (1, 1, self.all_args.recurrent_N, self.all_args.hidden_size),
            device=self.device
        )
        rnn_states_critic = torch.zeros(
            (1, 1, self.all_args.recurrent_N, self.all_args.hidden_size),
            device=self.device
        )
        masks = torch.ones((1, 1, 1), device=self.device)

        # If your rMAPPOPolicy's get_actions signature is:
        # get_actions(cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None, deterministic=False)
        # then do:
        values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = self.policy.get_actions(
            cent_obs=obs_tensor,  # or None if you have separate central obs
            obs=obs_tensor,
            rnn_states_actor=rnn_states_actor,
            rnn_states_critic=rnn_states_critic,
            masks=masks,
            available_actions=None,
            deterministic=False
        )

        action_np = _t2n(actions).squeeze()  # shape=()
        return action_np

    def train(self):
        """
        Minimal train placeholder: calls trainer.prep_training() then does a dummy update.
        (Real usage: gather rollouts in a buffer, call trainer.train(buffer), etc.)
        """
        self.trainer.prep_training()
        # dummy loss
        loss_val = np.random.random()
        return {"loss": loss_val}

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
