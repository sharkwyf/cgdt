"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import os
import sys
import argparse
import pickle
import random
import time
import gym
import d4rl
import torch
import numpy as np
import wandb

import utils
from copy import deepcopy
from tqdm import tqdm
from replay_buffer import ReplayBuffer
from lamb import Lamb
from stable_baselines3.common.vec_env import SubprocVecEnv
from pathlib import Path
from data import create_dataloader, discount_cumsum
from decision_transformer.models.decision_transformer import DecisionTransformer, ImitationTransformer
from evaluation import create_vec_eval_episodes_fn, vec_evaluate_episode_rtg
from trainer import SequenceTrainer, CriticSequenceTrainer, PretrainCriticSequenceTrainer
from logger import Logger

MAX_EPISODE_LEN = 1000
NUM_PASSED_STEPS = 20


class Experiment:
    def __init__(self, variant):

        self.state_dim, self.act_dim, self.action_range, self.discrete_action = self._get_env_spec(variant)
        self.gamma = variant["gamma"]
        self.offline_trajs, self.state_mean, self.state_std = self._load_dataset(
            master_work_dir, variant["env"], variant["no_reward"], variant["delayed_reward"]
        )

        top_data = self.offline_trajs[-int(len(self.offline_trajs) * variant["critic_top_percent"]):]
        train_size = int(0.9 * len(top_data))
        test_size = len(top_data) - train_size
        self.train_offline_trajs, self.test_offline_trajs = torch.utils.data.random_split(top_data, [train_size, test_size])

        # initialize by offline trajs
        self.replay_buffer = ReplayBuffer(variant["replay_size"], self.offline_trajs)

        self.aug_trajs = []

        self.device = variant.get("device", "cuda")
        self.target_entropy = -self.act_dim
        self.model = DecisionTransformer(
            state_dim=self.state_dim,
            act_dim=self.act_dim,
            action_range=self.action_range,
            max_length=variant["K"],
            eval_context_length=variant["eval_context_length"],
            max_ep_len=MAX_EPISODE_LEN,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
            n_head=variant["n_head"],
            n_inner=4 * variant["embed_dim"],
            activation_function=variant["activation_function"],
            n_positions=1024,
            resid_pdrop=variant["dropout"],
            attn_pdrop=variant["dropout"],
            stochastic_policy=True,
            discrete_action=self.discrete_action,
            ordering=variant["ordering"],
            init_temperature=variant["init_temperature"],
            target_entropy=self.target_entropy,
        ).to(device=self.device)

        self.critic = ImitationTransformer(
            state_dim=self.state_dim,
            act_dim=self.act_dim,
            action_range=self.action_range,
            max_length=variant["K"],
            eval_context_length=variant["eval_context_length"],
            max_ep_len=MAX_EPISODE_LEN,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_critic_layer"],
            n_head=variant["n_head"],
            n_inner=4 * variant["embed_dim"],
            activation_function=variant["activation_function"],
            n_positions=1024,
            resid_pdrop=variant["dropout"],
            attn_pdrop=variant["dropout"],
            stochastic_policy=True,
            discrete_action=self.discrete_action,
            ordering=variant["ordering"],
            init_temperature=variant["init_temperature"],
            target_entropy=self.target_entropy,
        ).to(device=self.device)

        self.critic_optimizer = Lamb(
            self.critic.parameters(),
            lr=variant["learning_rate"],
            weight_decay=variant["weight_decay"],
            eps=1e-8,
        )
        self.optimizer = Lamb(
            self.model.parameters(),
            lr=variant["learning_rate"],
            weight_decay=variant["weight_decay"],
            eps=1e-8,
        )
        self.pretrain_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.critic_optimizer, lambda steps: min((steps + 1) / variant["warmup_steps"], 1)
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda steps: min((steps + 1) / variant["warmup_steps"], 1)
        )

        self.log_temperature_optimizer = torch.optim.Adam(
            [self.model.log_temperature],
            lr=1e-4,
            betas=[0.9, 0.999],
        )

        # track the training progress and
        # training/evaluation/online performance in all the iterations

        self.best_pretrain_iter = None
        self.best_critic_state_dict = None
        self.min_pretrain_loss_mean = 1e9
        self.n_ascending_cnt = 0

        self.best_train_iter = None
        self.best_model_state_dict = None
        self.max_d4rl_score = -1e9

        self.pretrain_iter = 0
        self.train_iter = 0
        self.online_iter = 0
        self.total_transitions_sampled = 0
        self.variant = variant
        self.reward_scale = 1.0 if "antmaze" in variant["env"] or "kitchen" in variant["env"] or "bandit" in variant["env"] else 0.001
        if not variant["no_wandb"]:
            name = "{}{}_{}_rtg_{}_{}_iters_{}_{}_{}".format(
                variant["env"], "_delayed" if variant["delayed_reward"] else "", variant["tag"], variant["eval_rtg"], variant["online_rtg"], variant["max_pretrain_iters"], variant["max_train_iters"], variant["max_online_iters"]
            )
            variant["exp_name"] = name
            wandb.init(
                name=name,
                project="qdt-v4",
                config=variant,
                tags=[],
                reinit=True,
            )
            print(f"wandb initialized")
        self.logger = Logger(variant)

    def _get_env_spec(self, variant):
        if "bernoulli-bandit" in variant["env"]:
            from decision_transformer.envs.bernoulli_bandit import BernoulliBanditEnv
            env = BernoulliBanditEnv(
                num_arms=2,
                reward_power=3.0,
                reward_scale=0.9,
                generation_seed=0,
                bernoulli_prob=0.9,
                loop=False,
            )
            state_dim = 1
            act_dim = 2
            action_range = [0, 1]
            discrete_action = True
            env.close()
        else:
            env = gym.make(variant["env"])
            state_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
            action_range = [
                float(env.action_space.low.min()) + 1e-6,
                float(env.action_space.high.max()) - 1e-6,
            ]
            discrete_action = False
            env.close()
        return state_dim, act_dim, action_range, discrete_action

    def _save_model(self, path_prefix, is_pretrain_model=False):
        to_save = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "pretrain_iter": self.pretrain_iter,
            "train_iter": self.train_iter,
            "online_iter": self.online_iter,
            "args": self.variant,
            "total_transitions_sampled": self.total_transitions_sampled,
            "np": np.random.get_state(),
            "python": random.getstate(),
            "pytorch": torch.get_rng_state(),
            "log_temperature_optimizer_state_dict": self.log_temperature_optimizer.state_dict(),
            "best_train_iter": self.best_train_iter,
            "best_model_state_dict": self.best_model_state_dict,
            "max_d4rl_score": self.max_d4rl_score,
        }
        if self.critic is not None:
            to_save.update({
                "critic_state_dict": self.critic.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "pretrain_scheduler_state_dict": self.pretrain_scheduler.state_dict(),
                "best_critic_state_dict": self.best_critic_state_dict,
                "min_pretrain_loss_mean": self.min_pretrain_loss_mean,
                "n_ascending_cnt": self.n_ascending_cnt,
                "best_pretrain_iter": self.best_pretrain_iter,
            })

        with open(f"{path_prefix}/model.pt", "wb") as f:
            torch.save(to_save, f)
        print(f"\nModel saved at {path_prefix}/model.pt")

        if is_pretrain_model:
            with open(f"{path_prefix}/pretrain_model.pt", "wb") as f:
                torch.save(to_save, f)
            print(f"Model saved at {path_prefix}/pretrain_model.pt")

    def _load_model(self, path_prefix):
        if Path(f"{path_prefix}/model.pt").exists():
            with open(f"{path_prefix}/model.pt", "rb") as f:
                checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "critic_state_dict" in checkpoint:
                self.critic.load_state_dict(checkpoint["critic_state_dict"])
                self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
                self.pretrain_scheduler.load_state_dict(checkpoint["pretrain_scheduler_state_dict"])
                self.best_pretrain_iter = checkpoint["best_pretrain_iter"]
                self.best_critic_state_dict = checkpoint["best_critic_state_dict"]
                self.min_pretrain_loss_mean = checkpoint["min_pretrain_loss_mean"]
                self.n_ascending_cnt = checkpoint["n_ascending_cnt"]
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.log_temperature_optimizer.load_state_dict(
                checkpoint["log_temperature_optimizer_state_dict"]
            )

            self.best_train_iter = checkpoint["best_train_iter"]
            self.best_model_state_dict = checkpoint["best_model_state_dict"]
            self.max_d4rl_score = checkpoint["max_d4rl_score"]

            self.pretrain_iter = checkpoint["pretrain_iter"]
            self.train_iter = checkpoint["train_iter"]
            self.online_iter = checkpoint["online_iter"]
            self.total_transitions_sampled = checkpoint["total_transitions_sampled"]
            np.random.set_state(checkpoint["np"])
            random.setstate(checkpoint["python"])
            torch.set_rng_state(checkpoint["pytorch"])
            print(f"Model loaded at {path_prefix}/model.pt")

    def _load_dataset(self, work_dir, env_name, no_reward=False, delayed_reward=False):

        dataset_path = f"{work_dir}/data/{env_name}.pkl"
        with open(dataset_path, "rb") as f:
            trajectories = pickle.load(f)

        if no_reward:
            for traj in trajectories:
                traj["rewards"].fill(0)
        elif delayed_reward:
            for traj in trajectories:
                traj["rewards"][-1] = traj["rewards"].sum()
                traj["rewards"][:-1] = 0

        states, traj_lens, returns = [], [], []
        for path in trajectories:
            states.append(path["observations"])
            traj_lens.append(len(path["observations"]))
            returns.append(path["rewards"].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        # used for input normalization
        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        num_timesteps = sum(traj_lens)

        print("=" * 50)
        print(f"Starting new experiment: {env_name}")
        print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
        print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
        print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
        print(f"Average length: {np.mean(traj_lens):.2f}, std: {np.std(traj_lens):.2f}")
        print(f"Max length: {np.max(traj_lens):.2f}, min: {np.min(traj_lens):.2f}")
        print("=" * 50)

        sorted_inds = np.argsort(returns)  # lowest to highest
        num_trajectories = 1
        timesteps = traj_lens[sorted_inds[-1]]
        ind = len(trajectories) - 2
        while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
            timesteps += traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        sorted_inds = sorted_inds[-num_trajectories:]
        trajectories = [self._process_trajectory(trajectories[ii]) for ii in sorted_inds]

        return trajectories, state_mean, state_std

    def _process_trajectory(self, traj):
        traj["rtg"] = discount_cumsum(traj["rewards"], gamma=1.0)
        if self.gamma == 1.0:
            traj["rtg_discounted"] = traj["rtg"].copy()
        else:
            traj["rtg_discounted"] = discount_cumsum(traj["rewards"], gamma=self.gamma)
        return traj

    def _augment_trajectories(
        self,
        online_envs,
        target_explore,
        n,
        randomized=False,
    ):

        max_ep_len = MAX_EPISODE_LEN

        with torch.no_grad():
            # generate init state
            target_return = [target_explore * self.reward_scale] * online_envs.num_envs

            returns, lengths, trajs = vec_evaluate_episode_rtg(
                online_envs,
                self.state_dim,
                self.act_dim,
                self.model,
                max_ep_len=max_ep_len,
                reward_scale=self.reward_scale,
                target_return=target_return,
                mode="normal",
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=False,
                delayed_reward=self.variant["delayed_reward"],
            )

            trajs = [self._process_trajectory(traj) for traj in trajs]

        self.replay_buffer.add_new_trajs(trajs)
        self.aug_trajs += trajs
        self.total_transitions_sampled += np.sum(lengths)

        return {
            "aug_traj/return": np.mean(returns),
            "aug_traj/length": np.mean(lengths),
        }

    def pretrain(self, eval_envs, loss_fn, critic_loss_fn=None, value_loss_fn=None):
        print("\n\n\n*** Pretrain ***")

        eval_fns = [
            create_vec_eval_episodes_fn(
                env_name=self.variant["env"],
                vec_env=eval_envs,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,
                reward_scale=self.reward_scale,
                no_reward=self.variant["no_reward"],
                delayed_reward=self.variant["delayed_reward"],
            )
        ]

        if self.variant["trainer"] == "PretrainCriticSequenceTrainer":
            trainer = PretrainCriticSequenceTrainer(
                model=self.model,
                optimizer=self.optimizer,
                critic=self.critic,
                critic_optimizer=self.critic_optimizer,
                log_temperature_optimizer=self.log_temperature_optimizer,
                value_coef=self.variant["value_coef"],
                linear_value_coef_start=self.variant["linear_value_coef_start"],
                linear_value_coef_steps=self.variant["linear_value_coef_steps"],
                pretrain_scheduler=self.pretrain_scheduler,
                scheduler=self.scheduler,
                device=self.device,
                action_space=eval_envs.action_space,
            )
        elif self.variant["trainer"] == "CriticSequenceTrainer":
            trainer = CriticSequenceTrainer(
                model=self.model,
                optimizer=self.optimizer,
                critic=self.critic,
                critic_optimizer=self.critic_optimizer,
                log_temperature_optimizer=self.log_temperature_optimizer,
                value_coef=self.variant["value_coef"],
                scheduler=self.scheduler,
                device=self.device,
            )
        else:
            trainer = SequenceTrainer(
                model=self.model,
                optimizer=self.optimizer,
                log_temperature_optimizer=self.log_temperature_optimizer,
                scheduler=self.scheduler,
                device=self.device,
            )
        print(f"Pretraining Trainer: {type(trainer)}")

        while self.pretrain_iter < self.variant["max_pretrain_iters"]:
            # in every iteration, prepare the data loader
            train_critic_dataloader, test_critic_dataloader, train_dataloader = [None] * 3
            if self.n_ascending_cnt < NUM_PASSED_STEPS:
                train_critic_dataloader = create_dataloader(
                    trajectories=self.train_offline_trajs,
                    num_iters=self.variant["num_updates_per_critic_iter"],
                    batch_size=self.variant["batch_size"],
                    max_len=self.variant["K"],
                    state_dim=self.state_dim,
                    act_dim=self.act_dim,
                    state_mean=self.state_mean,
                    state_std=self.state_std,
                    reward_scale=self.reward_scale,
                    gamma=self.variant["gamma"],
                    action_range=self.action_range,
                    num_workers=self.variant["num_workers"],
                )

                test_critic_dataloader = create_dataloader(
                    trajectories=self.test_offline_trajs,
                    num_iters=self.variant["num_tests_per_critic_iter"],
                    batch_size=self.variant["batch_size"],
                    max_len=self.variant["K"],
                    state_dim=self.state_dim,
                    act_dim=self.act_dim,
                    state_mean=self.state_mean,
                    state_std=self.state_std,
                    reward_scale=self.reward_scale,
                    gamma=self.variant["gamma"],
                    action_range=self.action_range,
                    num_workers=self.variant["num_workers"],
                )

            if self.variant["num_updates_per_pretrain_iter"]:
                train_dataloader = create_dataloader(
                    trajectories=self.offline_trajs,
                    num_iters=self.variant["num_updates_per_pretrain_iter"],
                    batch_size=self.variant["batch_size"],
                    max_len=self.variant["K"],
                    state_dim=self.state_dim,
                    act_dim=self.act_dim,
                    state_mean=self.state_mean,
                    state_std=self.state_std,
                    reward_scale=self.reward_scale,
                    gamma=self.variant["gamma"],
                    action_range=self.action_range,
                    num_workers=self.variant["num_workers"],
                )

            train_outputs = trainer.pretrain_iteration(
                loss_fn=loss_fn,
                critic_loss_fn=critic_loss_fn,
                train_critic_dataloader=train_critic_dataloader,
                test_critic_dataloader=test_critic_dataloader,
                train_dataloader=train_dataloader,
            )
            if self.variant["num_updates_per_pretrain_iter"]:
                eval_outputs, eval_reward = self.evaluate(eval_fns)
            else:
                eval_outputs = {}
            outputs = {"time/total": time.time() - self.start_time}
            outputs.update(train_outputs)
            outputs.update(eval_outputs)
            self.logger.log_metrics(
                outputs,
                iter_num=self.pretrain_iter,
                total_transitions_sampled=self.total_transitions_sampled,
            )

            self._save_model(
                path_prefix=self.logger.log_path,
                is_pretrain_model=True,
            )

            if self.n_ascending_cnt < NUM_PASSED_STEPS:
                if outputs["training/eval_critic_loss_mean"] <= self.min_pretrain_loss_mean:
                    self.best_critic_state_dict = deepcopy(self.critic.state_dict())
                    self.best_pretrain_iter = self.pretrain_iter
                if outputs["training/eval_critic_loss_mean"] > self.min_pretrain_loss_mean:
                    self.n_ascending_cnt += 1
                else:
                    self.n_ascending_cnt = 0
                self.min_pretrain_loss_mean = min(outputs["training/eval_critic_loss_mean"], self.min_pretrain_loss_mean)

            self.pretrain_iter += 1

        if self.best_pretrain_iter is not None:
            print(f"\n\n\n*** Loading best critic from iter {self.best_pretrain_iter} ***")
            self.critic.load_state_dict(self.best_critic_state_dict)

        return outputs


    def train(self, eval_envs, loss_fn, critic_loss_fn=None, value_loss_fn=None):
        print("\n\n\n*** Train ***")

        eval_fns = [
            create_vec_eval_episodes_fn(
                env_name=self.variant["env"],
                vec_env=eval_envs,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,
                reward_scale=self.reward_scale,
                no_reward=self.variant["no_reward"],
                delayed_reward=self.variant["delayed_reward"],
            )
        ]

        if self.variant["trainer"] == "PretrainCriticSequenceTrainer":
            trainer = PretrainCriticSequenceTrainer(
                model=self.model,
                optimizer=self.optimizer,
                critic=self.critic,
                critic_optimizer=self.critic_optimizer,
                log_temperature_optimizer=self.log_temperature_optimizer,
                value_coef=self.variant["value_coef"],
                linear_value_coef_start=self.variant["linear_value_coef_start"],
                linear_value_coef_steps=self.variant["linear_value_coef_steps"],
                pretrain_scheduler=self.pretrain_scheduler,
                scheduler=self.scheduler,
                device=self.device,
                action_space=eval_envs.action_space,
            )
        elif self.variant["trainer"] == "CriticSequenceTrainer":
            trainer = CriticSequenceTrainer(
                model=self.model,
                optimizer=self.optimizer,
                critic=self.critic,
                critic_optimizer=self.critic_optimizer,
                log_temperature_optimizer=self.log_temperature_optimizer,
                value_coef=self.variant["value_coef"],
                scheduler=self.scheduler,
                device=self.device,
            )
        else:
            trainer = SequenceTrainer(
                model=self.model,
                optimizer=self.optimizer,
                log_temperature_optimizer=self.log_temperature_optimizer,
                scheduler=self.scheduler,
                device=self.device,
            )
        print(f"Training Trainer: {type(trainer)}")

        while self.train_iter < self.variant["max_train_iters"]:
            # in every iteration, prepare the data loader
            dataloader = create_dataloader(
                trajectories=self.offline_trajs,
                num_iters=self.variant["num_updates_per_train_iter"],
                batch_size=self.variant["batch_size"],
                max_len=self.variant["K"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                reward_scale=self.reward_scale,
                gamma=self.variant["gamma"],
                action_range=self.action_range,
                num_workers=self.variant["num_workers"],
            )

            train_outputs = trainer.train_iteration(
                loss_fn=loss_fn,
                critic_loss_fn=critic_loss_fn,
                value_loss_fn=value_loss_fn,
                dataloader=dataloader,
            )
            eval_outputs, eval_reward = self.evaluate(eval_fns)
            outputs = {"time/total": time.time() - self.start_time}
            outputs.update(train_outputs)
            outputs.update(eval_outputs)
            self.logger.log_metrics(
                outputs,
                iter_num=self.pretrain_iter + self.train_iter,
                total_transitions_sampled=self.total_transitions_sampled,
            )

            self._save_model(
                path_prefix=self.logger.log_path,
                is_pretrain_model=True,
            )

            if outputs["evaluation/score_mean_gm"] >= self.max_d4rl_score:
                self.best_train_iter = self.train_iter
                self.best_model_state_dict = deepcopy(self.model.state_dict())
                self.max_d4rl_score = outputs["evaluation/score_mean_gm"]
                print(f"Saving best model from iter {self.best_train_iter} ({self.max_d4rl_score}) ")

            self.train_iter += 1

        if self.best_train_iter is not None:
            print(f"\n\n\n*** Loading best model from iter {self.best_train_iter} ({self.max_d4rl_score}) ***")
            self.model.load_state_dict(self.best_model_state_dict)

        return outputs

    def online_tuning(self, online_envs, eval_envs, loss_fn, critic_loss_fn=None, value_loss_fn=None):

        print("\n\n\n*** Online Finetuning ***")

        if self.variant["use_critic"]:
            trainer = CriticSequenceTrainer(
                model=self.model,
                optimizer=self.optimizer,
                critic=self.critic,
                critic_optimizer=self.critic_optimizer,
                log_temperature_optimizer=self.log_temperature_optimizer,
                value_coef=self.variant["value_coef"],
                scheduler=self.scheduler,
                device=self.device,
            )
        else:
            trainer = SequenceTrainer(
                model=self.model,
                optimizer=self.optimizer,
                log_temperature_optimizer=self.log_temperature_optimizer,
                scheduler=self.scheduler,
                device=self.device,
            )
        print(f"Online Finetuning Trainer: {type(trainer)}")

        eval_fns = [
            create_vec_eval_episodes_fn(
                env_name=self.variant["env"],
                vec_env=eval_envs,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,
                reward_scale=self.reward_scale,
                no_reward=self.variant["no_reward"],
                delayed_reward=self.variant["delayed_reward"],
            )
        ]

        while self.online_iter < self.variant["max_online_iters"]:

            outputs = {}
            augment_outputs = self._augment_trajectories(
                online_envs,
                self.variant["online_rtg"],
                n=self.variant["num_online_rollouts"],
            )
            outputs.update(augment_outputs)

            dataloader = create_dataloader(
                trajectories=self.replay_buffer.trajectories,
                num_iters=self.variant["num_updates_per_online_iter"],
                batch_size=self.variant["batch_size"],
                max_len=self.variant["K"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                reward_scale=self.reward_scale,
                gamma=self.variant["gamma"],
                action_range=self.action_range,
                num_workers=self.variant["num_workers"],
            )

            # finetuning
            is_last_iter = self.online_iter == self.variant["max_online_iters"] - 1
            if (self.online_iter + 1) % self.variant[
                "eval_interval"
            ] == 0 or is_last_iter:
                evaluation = True
            else:
                evaluation = False

            train_outputs = trainer.train_iteration(
                loss_fn=loss_fn,
                critic_loss_fn=critic_loss_fn,
                value_loss_fn=value_loss_fn,
                dataloader=dataloader,
            )
            outputs.update(train_outputs)

            if evaluation:
                eval_outputs, eval_reward = self.evaluate(eval_fns)
                outputs.update(eval_outputs)

            outputs["time/total"] = time.time() - self.start_time

            # log the metrics
            self.logger.log_metrics(
                outputs,
                iter_num=self.pretrain_iter + self.train_iter + self.online_iter,
                total_transitions_sampled=self.total_transitions_sampled,
            )

            self._save_model(
                path_prefix=self.logger.log_path,
                is_pretrain_model=False,
            )

            self.online_iter += 1

        return outputs

    def final_evaluate(self, eval_envs):
        eval_start = time.time()
        self.model.eval()
        outputs = {}

        # evaluation on eval_rtg
        for eval_rtg_coef in [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
            eval_rtg = self.variant["eval_rtg"] * eval_rtg_coef
            print(f"Evaluating on eval_rtg_coef: {eval_rtg_coef}, ({eval_rtg})")
            eval_fns = [
                create_vec_eval_episodes_fn(
                    env_name=self.variant["env"],
                    vec_env=eval_envs,
                    eval_rtg=eval_rtg,
                    state_dim=self.state_dim,
                    act_dim=self.act_dim,
                    state_mean=self.state_mean,
                    state_std=self.state_std,
                    device=self.device,
                    use_mean=True,
                    reward_scale=self.reward_scale,
                    no_reward=self.variant["no_reward"],
                    delayed_reward=self.variant["delayed_reward"],
                )
            ] * int(100 / self.variant["num_eval_rollouts"])
            score_mean_gms = []
            for eval_fn in eval_fns:
                o = eval_fn(self.model)
                score_mean_gms.append(o["evaluation/score_mean_gm"])
            outputs[f"final_evaluation/score_mean_gm_rtg{eval_rtg_coef}"] = np.mean(score_mean_gms)

        # evaluation on eval_context_len
        eval_fns = [
            create_vec_eval_episodes_fn(
                env_name=self.variant["env"],
                vec_env=eval_envs,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,
                reward_scale=self.reward_scale,
                no_reward=self.variant["no_reward"],
                delayed_reward=self.variant["delayed_reward"],
            )
        ] * int(100 / self.variant["num_eval_rollouts"])
        for eval_context_length in [1, 5, 10, 15, 20]:
            print(f"Evaluating on eval_context_length: {eval_context_length}")
            if eval_context_length > self.variant["K"]:
                break
            self.model.eval_context_length = eval_context_length
            score_mean_gms = []
            for eval_fn in eval_fns:
                o = eval_fn(self.model)
                score_mean_gms.append(o["evaluation/score_mean_gm"])
            outputs[f"final_evaluation/score_mean_gm_ctx{eval_context_length}"] = np.mean(score_mean_gms)

        outputs["time/final_evaluation"] = time.time() - eval_start

        self.logger.log_metrics(
            outputs,
            iter_num=self.pretrain_iter + self.train_iter + self.online_iter,
            total_transitions_sampled=self.total_transitions_sampled,
        )

        return outputs

    def evaluate(self, eval_fns):
        eval_start = time.time()
        self.model.eval()
        outputs = {}
        for eval_fn in eval_fns:
            for _ in range(int(self.variant["num_eval_episodes"] / self.variant["num_eval_rollouts"])):
                o = eval_fn(self.model)
                for k, v in o.items():
                    if k not in outputs:
                        outputs[k] = [v]
                    else:
                        outputs[k].append(v)
        for k, v in outputs.items():
            outputs[k] = np.mean(v)
        outputs["time/evaluation"] = time.time() - eval_start

        eval_reward = outputs["evaluation/return_mean_gm"]
        return outputs, eval_reward

    def run(self):

        utils.set_seed_everywhere(args.seed)

        import d4rl

        def critic_loss_fn(
            rtg_hat,
            rtg,
            attention_mask,
            **kwargs,
        ):
            mse, nll, asymmetric_l2_loss = [torch.tensor(0, device=rtg.device) for _ in range(3)]
            if self.variant["critic_loss"] == "nll_infonce":
                mse = torch.nn.functional.mse_loss(rtg_hat.mu[attention_mask > 0], rtg[attention_mask > 0], reduction='mean')

                shuffled_rtg_hat = kwargs["shuffled_return_preds"]
                # pos_nll = (rtg_hat.nll(rtg)[attention_mask > 0] * (-1)).exp().sum().log()
                # neg_nll = (shuffled_rtg_hat.nll(rtg)[:, -1][attention_mask[:, -1] > 0] * (-1)).exp().sum().log()
                # loss = - pos_nll + neg_nll / rtg.shape[1] * 0

                nll = rtg_hat.nll(rtg)[attention_mask > 0].mean()
                shuffled_nll = shuffled_rtg_hat.nll(0)[:, -1][attention_mask[:, -1] > 0].mean()
                loss = nll + shuffled_nll / rtg.shape[1] * 1
                
                # Add noise for neg_nll
                # TODO: improve for all positions

            elif self.variant["critic_loss"] == "expectile":
                tau = self.variant["tau1"]
                mse = torch.nn.functional.mse_loss(rtg_hat.mu[attention_mask > 0], rtg[attention_mask > 0], reduction='mean')
                u = (rtg - rtg_hat.mu) / rtg_hat.sigma / self.variant["beta"]    # TODO: test with mean() or not
                nll = torch.mean((torch.abs(tau - (u < 0).float()).squeeze(-1) * rtg_hat.nll(rtg))[attention_mask > 0]) * (1 / max(tau, 1 - tau))
                loss = nll
            elif self.variant["critic_loss"] == "nll":
                mse = torch.nn.functional.mse_loss(rtg_hat.mu[attention_mask > 0], rtg[attention_mask > 0], reduction='mean')
                nll = rtg_hat.nll(rtg)[attention_mask > 0].mean()
                loss = nll
            elif self.variant["critic_loss"] == "mse":
                mse = torch.nn.functional.mse_loss(rtg_hat.mu[attention_mask > 0], rtg[attention_mask > 0], reduction='mean')
                mse = mse / rtg[attention_mask > 0].mean().detach() ** 2    # scale critic loss 
                nll = torch.tensor(0, device=rtg.device)
                loss = mse
            else:
                raise NotImplementedError()

            return (
                loss,
                mse,
                nll,
                rtg[attention_mask > 0],
                rtg_hat.mu[attention_mask > 0],
                rtg_hat.sigma[attention_mask > 0],
            )

        def value_loss_fn(
            rtg_hat,
            rtg,
            attention_mask,
        ):
            mse, nll, asymmetric_l2_loss = [torch.tensor(0, device=rtg.device) for _ in range(3)]
            if self.variant["value_loss"] == "expectile":
                u = (rtg - rtg_hat.mu) / rtg_hat.sigma / self.variant["beta"]
                asymmetric_l2_loss = torch.mean(torch.abs(self.variant["tau"] - (u < 0).float()) * u**2) \
                    * (1 / max(self.variant["tau"], 1 - self.variant["tau"]))
            elif self.variant["value_loss"] == "gumbel":
                u = (rtg - rtg_hat.mu) / rtg_hat.sigma / self.variant["beta"]
                asymmetric_l2_loss = torch.mean((-u).exp() + u)
            elif self.variant["value_loss"] == "nll":
                nll = rtg_hat.nll(rtg)[attention_mask > 0].mean()
            elif self.variant["value_loss"] == "mse":
                mse = torch.nn.functional.mse_loss(rtg_hat.mu[attention_mask > 0], rtg[attention_mask > 0], reduction='mean')
                mse = mse / rtg[attention_mask > 0].mean().detach() ** 2
            else:
                raise NotImplementedError()
            
            loss = mse + nll + asymmetric_l2_loss

            return (
                loss,
                mse,
                nll,
                rtg[attention_mask > 0],
                rtg_hat.mu[attention_mask > 0],
                rtg_hat.sigma[attention_mask > 0],
            )

        def loss_fn(
            a_hat_dist,
            a,
            attention_mask,
            entropy_reg,
        ):
            # a_hat is a SquashedNormal Distribution
            if self.variant["action_loss"] == "mse":
                if self.discrete_action:
                    a_hat = a_hat_dist
                else:
                    a_hat = a_hat_dist.mean
                mse_loss = torch.nn.functional.mse_loss(a_hat[attention_mask > 0], a[attention_mask > 0], reduction='mean')
                log_likelihood = torch.tensor(0, device=a.device)
            elif self.variant["action_loss"] == "nll":
                mse_loss = torch.tensor(0, device=a.device)
                log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()
            else:
                raise NotImplementedError()

            if self.variant["action_loss"] == "nll_entropy":
                entropy = a_hat_dist.entropy().mean()
            else:
                entropy = torch.tensor(self.target_entropy, device=a.device).detach()

            loss = -(log_likelihood + entropy_reg * entropy) + mse_loss

            return (
                loss,
                -log_likelihood,
                entropy,
                mse_loss,
            )

        def get_env_builder(seed, env_name, target_goal=None):
            if "bernoulli-bandit" in env_name:
                def make_env_fn():
                    from decision_transformer.envs.bernoulli_bandit import BernoulliBanditEnv, BernoulliBanditWrapper
                    bernoulli_prob = 1 - float(env_name[-3:])
                    env = BernoulliBanditEnv(
                        num_arms=2,
                        reward_power=3.0,
                        reward_scale=0.9,
                        generation_seed=0,
                        bernoulli_prob=bernoulli_prob,
                        loop=False,
                    )
                    env = BernoulliBanditWrapper(env)
                    env.seed(seed)
                    if hasattr(env.env, "wrapped_env"):
                        env.env.wrapped_env.seed(seed)
                    elif hasattr(env.env, "seed"):
                        env.env.seed(seed)
                    else:
                        pass
                    env.action_space.seed(seed)
                    env.observation_space.seed(seed)

                    if target_goal:
                        env.set_target_goal(target_goal)
                        print(f"Set the target goal to be {env.target_goal}")
                    
                    print(f"Make env {env_name} with bernoulli_prob: {bernoulli_prob}")
                    return env

                return make_env_fn
            else:
                def make_env_fn():
                    import d4rl

                    env = gym.make(env_name)
                    env.seed(seed)
                    if hasattr(env.env, "wrapped_env"):
                        env.env.wrapped_env.seed(seed)
                    elif hasattr(env.env, "seed"):
                        env.env.seed(seed)
                    else:
                        pass
                    env.action_space.seed(seed)
                    env.observation_space.seed(seed)

                    if target_goal:
                        env.set_target_goal(target_goal)
                        print(f"Set the target goal to be {env.target_goal}")
                    return env

                return make_env_fn

        print("\n\nMaking Eval Env.....")
        env_name = self.variant["env"]
        if "antmaze" in env_name:
            env = gym.make(env_name)
            target_goal = env.target_goal
            env.close()
            print(f"Generated the fixed target goal: {target_goal}")
        else:
            target_goal = None
        eval_envs = SubprocVecEnv(
            [
                get_env_builder(i, env_name=env_name, target_goal=target_goal)
                for i in range(self.variant["num_eval_rollouts"])
            ]
        )

        outputs = {}
        self.start_time = time.time()
        if self.variant["max_pretrain_iters"]:
            outputs = self.pretrain(eval_envs, loss_fn, critic_loss_fn, value_loss_fn)

        if self.variant["max_train_iters"]:
            outputs = self.train(eval_envs, loss_fn, critic_loss_fn, value_loss_fn)

        if self.variant["max_online_iters"]:
            print("\n\nMaking Online Env.....")
            online_envs = SubprocVecEnv(
                [
                    get_env_builder(i + 100, env_name=env_name, target_goal=target_goal)
                    for i in range(self.variant["num_online_rollouts"])
                ]
            )
            outputs = self.online_tuning(online_envs, eval_envs, loss_fn, critic_loss_fn, value_loss_fn)
            online_envs.close()

        if True:
            outputs = self.final_evaluate(eval_envs)

        eval_envs.close()

        if not self.variant["no_wandb"]:
            wandb.finish()
            print(f"wandb finialized")

        return outputs


def train(param):
    args = param["args"]
    for k in param.keys():
        if k not in ["args", "tasks"]:
            args.__setattr__(k, param[k])
    if "tasks" in param:
        for k in param["tasks"].keys():
            assert k in args, f"no {k} in args"
            args.__setattr__(k, param["tasks"][k])
            
    utils.set_seed_everywhere(args.seed)
    experiment = Experiment(vars(args))

    print("=" * 50)
    outputs = experiment.run()
    
    return {
        "max_d4rl_score": max([v for k, v in outputs.items() if "score_mean_gm" in k]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--env", type=str, default="hopper-medium-v2")
    parser.add_argument("--no_reward", action="store_true")
    parser.add_argument("--delayed_reward", action="store_true")

    # model options
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval_context_length", type=int, default=5)
    # 0: no pos embedding others: absolute ordering
    parser.add_argument("--ordering", type=int, default=0)

    # critic options
    parser.add_argument("--critic_top_percent", type=float, default=1.)
    parser.add_argument("--gamma", type=float, default=1.)
    parser.add_argument("--n_critic_layer", type=int, default=4)
    
    parser.add_argument("--trainer", type=str, choices=["CriticSequenceTrainer", "PretrainCriticSequenceTrainer"])
    parser.add_argument("--action_loss", type=str, choices=["mse", "nll", "nll_entropy"])
    parser.add_argument("--critic_loss", type=str, choices=["mse", "nll", "expectile", "nll_infonce"])
    parser.add_argument("--value_loss", type=str, choices=["mse", "nll", "expectile", "gumbel"])
    parser.add_argument("--value_coef", type=float, default=0.)
    parser.add_argument("--linear_value_coef_start", type=int, default=0)
    parser.add_argument("--linear_value_coef_steps", type=int, default=0)
    parser.add_argument("--beta", type=float, default=1.)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--tau1", type=float, default=0.5)

    # shared evaluation options
    parser.add_argument("--eval_rtg", type=int, default=3600)
    parser.add_argument("--num_eval_rollouts", type=int, default=10)
    parser.add_argument("--num_eval_episodes", type=int, default=10)

    # shared training options
    parser.add_argument("--init_temperature", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=5e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)

    # pretraining options
    parser.add_argument("--max_pretrain_iters", type=int, default=100)
    parser.add_argument("--num_updates_per_critic_iter", type=int, default=1200)
    parser.add_argument("--num_tests_per_critic_iter", type=int, default=200)
    parser.add_argument("--num_updates_per_pretrain_iter", type=int, default=0)

    # training options
    parser.add_argument("--max_train_iters", type=int, default=100)
    parser.add_argument("--num_updates_per_train_iter", type=int, default=800)

    # finetuning options
    parser.add_argument("--max_online_iters", type=int, default=1500)
    parser.add_argument("--online_rtg", type=int, default=7200)
    parser.add_argument("--num_online_rollouts", type=int, default=1)
    parser.add_argument("--replay_size", type=int, default=1000)
    parser.add_argument("--num_updates_per_online_iter", type=int, default=300)
    parser.add_argument("--eval_interval", type=int, default=10)

    # environment options
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_to_tb", "-w", type=bool, default=True)
    parser.add_argument("--save_dir", type=str, default="./exp")
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument('--no_wandb', action="store_true")

    args = parser.parse_args()

    train({
        "args": args,
    })
