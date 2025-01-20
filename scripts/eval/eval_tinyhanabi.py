#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from datetime import datetime

from onpolicy.config import get_config
from onpolicy.envs.tinyhanabi.Tiny_Hanabi_Env import TinyHanabiEnv


def make_eval_env(all_args):
    env = TinyHanabiEnv(all_args, all_args.seed)
    return env

def parse_args(args, parser):
    parser.add_argument('--hanabi_name', type=str, default='TinyHanabi-Example', help="Which tiny hanabi variant to run on")
    # parser.add_argument('--model_dir', type=str, required=True, help="Directory of the saved model")
    all_args = parser.parse_known_args(args)[0]
    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # Device setup
    if all_args.cuda and torch.cuda.is_available():
        print("Using GPU...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
    else:
        print("Using CPU...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # Run directory
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / all_args.env_name / all_args.hanabi_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb setup
    if all_args.use_wandb:
        current_time = datetime.now().strftime("%d_%H_%M")
        run = wandb.init(
            config=all_args,
            project=all_args.env_name,
            notes=socket.gethostname(),
            name=str(all_args.algorithm_name) + "_eval_" + str(all_args.experiment_name) + "_at_" + current_time,
            group=all_args.hanabi_name,
            dir=str(run_dir),
            job_type="evaluation",
            reinit=True
        )
    else:
        run = None

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-Eval-" + str(all_args.env_name) + "-" + str(all_args.experiment_name))

    # Seed setup
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # Env initialization
    eval_envs = make_eval_env(all_args)

    # Import the correct Runner
    if all_args.share_policy:
        from onpolicy.runner.shared.tinyhanabi_runner_forward import TinyHanabiRunner as Runner
    else:
        from onpolicy.runner.separated.tinyhanabi_runner_forward import TinyHanabiRunner as Runner

    # Config for Runner
    config = {
        "all_args": all_args,
        "envs": eval_envs,
        "eval_envs": eval_envs,
        "num_agents": 2,
        "device": device,
        "run_dir": run_dir
    }

    runner = Runner(config)
    runner.restore(all_args.model_dir)

    runner.eval_policy()

    eval_envs.close()
    if run:
        run.finish()

if __name__ == "__main__":
    main(sys.argv[1:])
