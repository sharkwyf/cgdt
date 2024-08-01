"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import torch
import time
import wandb


class SequenceTrainer:
    def __init__(
        self,
        model,
        optimizer,
        log_temperature_optimizer,
        scheduler=None,
        device="cuda",
    ):
        self.model = model
        self.optimizer = optimizer
        self.log_temperature_optimizer = log_temperature_optimizer
        self.scheduler = scheduler
        self.device = device
        self.start_time = time.time()

    def train_iteration(
        self,
        loss_fn,
        dataloader,
        **kwargs,
    ):

        losses, nlls, entropies, mses = [], [], [], []
        logs = dict()
        train_start = time.time()

        self.model.train()
        for _, trajs in enumerate(dataloader):
            loss, nll, entropy, mse = self.train_step_stochastic(loss_fn, trajs)
            losses.append(loss)
            nlls.append(nll)
            entropies.append(entropy)
            mses.append(mse)

        logs["time/training"] = time.time() - train_start
        logs["training/train_loss_mean"] = np.mean(losses)
        logs["training/train_loss_std"] = np.std(losses)
        logs["training/nll"] = nlls[-1]
        logs["training/entropy"] = entropies[-1]
        logs["training/mse"] = mses[-1]
        logs["training/temp_value"] = self.model.temperature().detach().cpu().item()

        return logs

    def train_step_stochastic(self, loss_fn, trajs):
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            rtg_discounted,
            timesteps,
            ordering,
            padding_mask,
        ) = trajs

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        rtg = rtg.to(self.device)
        timesteps = timesteps.to(self.device)
        ordering = ordering.to(self.device)
        padding_mask = padding_mask.to(self.device)

        action_target = torch.clone(actions)

        _, action_preds, _ = self.model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            ordering,
            padding_mask=padding_mask,
        )

        loss, nll, entropy, mse = loss_fn(
            action_preds,  # a_hat_dist
            action_target,
            padding_mask,
            self.model.temperature().detach(),  # no gradient taken here
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        self.log_temperature_optimizer.zero_grad()
        temperature_loss = (
            self.model.temperature() * (entropy - self.model.target_entropy).detach()
        )
        temperature_loss.backward()
        self.log_temperature_optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return (
            loss.detach().cpu().item(),
            nll.detach().cpu().item(),
            entropy.detach().cpu().item(),
            mse.detach().cpu().item(),
        )


class PretrainCriticSequenceTrainer:
    def __init__(
        self,
        model,
        optimizer,
        critic,
        critic_optimizer,
        log_temperature_optimizer,
        value_coef=0.,
        linear_value_coef_start=0,
        linear_value_coef_steps=0,
        pretrain_scheduler=None,
        scheduler=None,
        device="cuda",
        action_space=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.critic = critic
        self.critic_optimizer = critic_optimizer
        self.log_temperature_optimizer = log_temperature_optimizer
        self.max_value_coef = value_coef
        self.linear_value_coef_start = linear_value_coef_start
        self.linear_value_coef_steps = linear_value_coef_steps
        self.pretrain_scheduler = pretrain_scheduler
        self.scheduler = scheduler
        self.device = device
        self.action_space = action_space
        self.start_time = time.time()
        self.step = 0

    @property
    def value_coef(self):
        step = max(self.step - self.linear_value_coef_start, 0)
        return min((step + 1) / max(1, self.linear_value_coef_steps) * self.max_value_coef, self.max_value_coef)

    def pretrain_iteration(
        self,
        loss_fn,
        critic_loss_fn,
        train_critic_dataloader,
        test_critic_dataloader,
        train_dataloader,
        **kwargs,
    ):
        logs = dict()
        train_start = time.time()

        # Pretraining Critic
        if train_critic_dataloader is not None:
            critic_losses, critic_mses, critic_nlls, rtgs, rtg_hat_means, rtg_hat_stds = [], [], [], [], [], []
            eval_critic_losses, eval_critic_mses, eval_critic_nlls, eval_rtgs, eval_rtg_hat_means, eval_rtg_hat_stds = [], [], [], [], [], []

            self.critic.train()
            for _, trajs in enumerate(train_critic_dataloader):
                critic_loss, critic_mse, critic_nll, rtg, rtg_hat_mean, rtg_hat_std = self.train_step_critic(critic_loss_fn, trajs)
                critic_losses.append(critic_loss)
                critic_mses.append(critic_mse)
                critic_nlls.append(critic_nll)
                rtgs.append(rtg)
                rtg_hat_means.append(rtg_hat_mean)
                rtg_hat_stds.append(rtg_hat_std)

            self.critic.eval()
            with torch.no_grad():
                for _, trajs in enumerate(test_critic_dataloader):
                    critic_loss, critic_mse, critic_nll, rtg, rtg_hat_mean, rtg_hat_std = self.train_step_critic(critic_loss_fn, trajs, evaluate=True)
                    eval_critic_losses.append(critic_loss)
                    eval_critic_mses.append(critic_mse)
                    eval_critic_nlls.append(critic_nll)
                    eval_rtgs.append(rtg)
                    eval_rtg_hat_means.append(rtg_hat_mean)
                    eval_rtg_hat_stds.append(rtg_hat_std)

            logs["training/train_critic_loss_mean"] = np.mean(critic_losses)
            logs["training/train_critic_loss_std"] = np.std(critic_losses)
            logs["training/train_critic_mse_mean"] = np.mean(critic_mses)
            logs["training/train_critic_mse_std"] = np.std(critic_mses)
            logs["training/train_critic_nll_mean"] = np.mean(critic_nlls)
            logs["training/train_critic_nll_std"] = np.std(critic_nlls)
            logs["training/train_rtg_mean"] = np.mean(np.concatenate(rtgs))
            logs["training/train_rtg_hat_mean"] = np.mean(np.concatenate(rtg_hat_means))
            logs["training/train_rtg_hat_std"] = np.mean(np.concatenate(rtg_hat_stds))
            logs["training/train_rtg_delta_std"] = np.std(np.concatenate(rtg_hat_means) - np.concatenate(rtgs))
            logs["training/eval_critic_loss_mean"] = np.mean(eval_critic_losses)
            logs["training/eval_critic_loss_std"] = np.std(eval_critic_losses)
            logs["training/eval_critic_mse_mean"] = np.mean(eval_critic_mses)
            logs["training/eval_critic_mse_std"] = np.std(eval_critic_mses)
            logs["training/eval_critic_nll_mean"] = np.mean(eval_critic_nlls)
            logs["training/eval_critic_nll_std"] = np.std(eval_critic_nlls)
            logs["training/eval_rtg_mean"] = np.mean(np.concatenate(eval_rtgs))
            logs["training/eval_rtg_hat_mean"] = np.mean(np.concatenate(eval_rtg_hat_means))
            logs["training/eval_rtg_hat_std"] = np.mean(np.concatenate(eval_rtg_hat_stds))
            logs["training/eval_rtg_delta_std"] = np.std(np.concatenate(eval_rtg_hat_means) - np.concatenate(eval_rtgs))

            a_max = max(np.concatenate(rtgs).max(), np.concatenate(eval_rtgs).max())
            logs["training/rtg_mean_histogram"] = wandb.Histogram(np.concatenate(rtgs), num_bins=512)
            logs["training/rtg_hat_delta_histogram"] = wandb.Histogram(np.clip(np.concatenate(rtg_hat_means) - np.concatenate(rtgs), a_min=-a_max / 2, a_max=a_max / 2), num_bins=512)
            logs["training/rtg_hat_mean_histogram"] = wandb.Histogram(np.clip(np.concatenate(rtg_hat_means), a_min=0, a_max=a_max), num_bins=512)
            logs["training/rtg_hat_std_histogram"] = wandb.Histogram(np.clip(np.concatenate(rtg_hat_stds), a_min=0, a_max=a_max), num_bins=512)
            logs["training/eval_rtg_mean_histogram"] = wandb.Histogram(np.concatenate(eval_rtgs), num_bins=512)
            logs["training/eval_rtg_hat_delta_histogram"] = wandb.Histogram(np.clip(np.concatenate(eval_rtg_hat_means) - np.concatenate(eval_rtgs), a_min=-a_max / 2, a_max=a_max / 2), num_bins=512)
            logs["training/eval_rtg_hat_mean_histogram"] = wandb.Histogram(np.clip(np.concatenate(eval_rtg_hat_means), a_min=0, a_max=a_max), num_bins=512)
            logs["training/eval_rtg_hat_std_histogram"] = wandb.Histogram(np.clip(np.concatenate(eval_rtg_hat_stds), a_min=0, a_max=a_max), num_bins=512)

        # Pretraining Policy
        if train_dataloader is not None:
            losses, nlls, entropies, mses = [], [], [], []

            self.model.train()
            for _, trajs in enumerate(train_dataloader):
                loss, nll, entropy, mse = self.train_step_stochastic(loss_fn, trajs)
                losses.append(loss)
                nlls.append(nll)
                entropies.append(entropy)
                mses.append(mse)

            logs["training/train_loss_mean"] = np.mean(losses)
            logs["training/train_loss_std"] = np.std(losses)
            logs["training/nll"] = nlls[-1]
            logs["training/entropy"] = entropies[-1]
            logs["training/mse"] = mses[-1]
            logs["training/temp_value"] = self.model.temperature().detach().cpu().item()

        logs["time/training"] = time.time() - train_start
        return logs

    def train_iteration(
        self,
        loss_fn,
        critic_loss_fn,
        value_loss_fn,
        dataloader,
    ):

        losses, nlls, entropies, mses = [], [], [], []
        critic_losses, value_losses = [], []
        rtgs, rtg_hat_means, rtg_hat_stds = [], [], []
        logs = dict()
        train_start = time.time()

        self.model.train()
        self.critic.eval()
        for _, trajs in enumerate(dataloader):
            loss, nll, entropy, mse, value_loss, rtg, rtg_hat_mean, rtg_hat_std = self.train_step_stochastic_with_value(loss_fn, value_loss_fn, trajs)
            losses.append(loss)
            nlls.append(nll)
            entropies.append(entropy)
            mses.append(mse)
            value_losses.append(value_loss)
            rtgs.append(rtg)
            rtg_hat_means.append(rtg_hat_mean)
            rtg_hat_stds.append(rtg_hat_std)

        logs["time/training"] = time.time() - train_start
        logs["training/train_loss_mean"] = np.mean(losses)
        logs["training/train_loss_std"] = np.std(losses)
        logs["training/nll"] = nlls[-1]
        logs["training/entropy"] = entropies[-1]
        logs["training/mse"] = mses[-1]
        logs["training/train_value_loss_mean"] = np.mean(value_losses)
        logs["training/train_value_loss_std"] = np.std(value_losses)
        logs["training/train_rtg_mean"] = np.mean(np.concatenate(rtgs))
        logs["training/train_rtg_hat_mean"] = np.mean(np.concatenate(rtg_hat_means))
        logs["training/train_rtg_hat_std"] = np.mean(np.concatenate(rtg_hat_stds))
        logs["training/temp_value"] = self.model.temperature().detach().cpu().item()
        logs["training/value_coef"] = self.value_coef
        
        a_max = np.concatenate(rtgs).max()
        logs["training/rtg_mean_histogram"] = wandb.Histogram(np.concatenate(rtgs), num_bins=512)
        logs["training/rtg_hat_delta_histogram"] = wandb.Histogram(np.clip(np.concatenate(rtg_hat_means) - np.concatenate(rtgs), a_min=-a_max / 2, a_max=a_max / 2), num_bins=512)
        logs["training/rtg_hat_mean_histogram"] = wandb.Histogram(np.clip(np.concatenate(rtg_hat_means), a_min=0, a_max=a_max), num_bins=512)
        logs["training/rtg_hat_std_histogram"] = wandb.Histogram(np.clip(np.concatenate(rtg_hat_stds), a_min=0, a_max=a_max), num_bins=512)

        return logs

    def train_step_critic(self, critic_loss_fn, trajs, evaluate=False):
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            rtg_discounted,
            timesteps,
            ordering,
            padding_mask,
        ) = trajs

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        rtg_discounted = rtg_discounted.to(self.device)
        timesteps = timesteps.to(self.device)
        ordering = ordering.to(self.device)
        padding_mask = padding_mask.to(self.device)

        rtg_discounted_target = torch.clone(rtg_discounted)

        _, _, return_preds = self.critic.forward(
            states,
            actions,
            rewards,
            rtg_discounted[:, :-1],
            timesteps,
            ordering,
            padding_mask=padding_mask,
        )

        if False:
            neg_actions = torch.clone(actions)
            shuffle_indices = torch.randperm(actions.shape[0])
            # neg_actions[:, -1] = neg_actions[shuffle_indices, -1]
            neg_actions[:, -1] = torch.tensor(np.stack([self.action_space.sample() for _ in range(neg_actions.shape[0])]), device=neg_actions.device)

            _, _, shuffled_return_preds = self.critic.forward(
                states,
                neg_actions,
                rewards,
                rtg_discounted[:, :-1],
                timesteps,
                ordering,
                padding_mask=padding_mask,
            )
        else:
            shuffled_return_preds = None

        critic_loss, critic_mse, critic_nll, rtg, rtg_hat_mean, rtg_hat_std = critic_loss_fn(
            return_preds,
            rtg_discounted_target[:, :-1],
            padding_mask,
            shuffled_return_preds=shuffled_return_preds,
        )

        if not evaluate:
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.25)
            self.critic_optimizer.step()

            if self.pretrain_scheduler is not None:
                self.pretrain_scheduler.step()

        return (
            critic_loss.detach().cpu().item(),
            critic_mse.detach().cpu().item(),
            critic_nll.detach().cpu().item(),
            rtg.detach().cpu().numpy(),
            rtg_hat_mean.detach().cpu().numpy(),
            rtg_hat_std.detach().cpu().numpy(),
        )

    def train_step_stochastic(self, loss_fn, trajs):
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            rtg_discounted,
            timesteps,
            ordering,
            padding_mask,
        ) = trajs

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        rtg = rtg.to(self.device)
        timesteps = timesteps.to(self.device)
        ordering = ordering.to(self.device)
        padding_mask = padding_mask.to(self.device)

        action_target = torch.clone(actions)

        _, action_preds, _ = self.model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            ordering,
            padding_mask=padding_mask,
        )

        loss, nll, entropy, mse = loss_fn(
            action_preds,  # a_hat_dist
            action_target,
            padding_mask,
            self.model.temperature().detach(),  # no gradient taken here
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        self.log_temperature_optimizer.zero_grad()
        temperature_loss = (
            self.model.temperature() * (entropy - self.model.target_entropy).detach()
        )
        temperature_loss.backward()
        self.log_temperature_optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return (
            loss.detach().cpu().item(),
            nll.detach().cpu().item(),
            entropy.detach().cpu().item(),
            mse.detach().cpu().item(),
        )

    def train_step_stochastic_with_value(self, loss_fn, value_loss_fn, trajs):
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            rtg_discounted,
            timesteps,
            ordering,
            padding_mask,
        ) = trajs

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        rtg = rtg.to(self.device)
        rtg_discounted = rtg_discounted.to(self.device)
        timesteps = timesteps.to(self.device)
        ordering = ordering.to(self.device)
        padding_mask = padding_mask.to(self.device)

        action_target = torch.clone(actions)
        rtg_discounted_target = torch.clone(rtg_discounted)

        _, action_preds, _ = self.model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            ordering,
            padding_mask=padding_mask,
        )

        loss, nll, entropy, mse = loss_fn(
            action_preds,  # a_hat_dist
            action_target,
            padding_mask,
            self.model.temperature().detach(),  # no gradient taken here
        )

        _, _, return_preds = self.critic.forward(
            states,
            action_preds if self.model.discrete_action else action_preds.mean,
            rewards,
            rtg_discounted[:, :-1],
            timesteps,
            ordering,
            padding_mask=padding_mask,
        )

        value_loss, value_mse, value_nll, rtg, rtg_hat_mean, rtg_hat_std = value_loss_fn(
            return_preds,
            rtg_discounted_target[:, :-1],
            padding_mask,
        )

        self.optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        ((loss + value_loss * self.value_coef) / np.abs(1 + self.value_coef)).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()
        self.step += 1

        self.log_temperature_optimizer.zero_grad()
        temperature_loss = (
            self.model.temperature() * (entropy - self.model.target_entropy).detach()
        )
        temperature_loss.backward()
        self.log_temperature_optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return (
            loss.detach().cpu().item(),
            nll.detach().cpu().item(),
            entropy.detach().cpu().item(),
            mse.detach().cpu().item(),
            value_loss.detach().cpu().item(),
            rtg.detach().cpu().numpy(),
            rtg_hat_mean.detach().cpu().numpy(),
            rtg_hat_std.detach().cpu().numpy(),
        )


class CriticSequenceTrainer:
    def __init__(
        self,
        model,
        optimizer,
        critic,
        critic_optimizer,
        log_temperature_optimizer,
        value_coef=0.,
        scheduler=None,
        device="cuda",
    ):
        self.model = model
        self.optimizer = optimizer
        self.critic = critic
        self.critic_optimizer = critic_optimizer
        self.log_temperature_optimizer = log_temperature_optimizer
        self.value_coef = value_coef
        self.scheduler = scheduler
        self.device = device
        self.start_time = time.time()

    def train_iteration(
        self,
        loss_fn,
        critic_loss_fn,
        value_loss_fn,
        dataloader,
    ):

        losses, nlls, entropies, mses = [], [], [], []
        critic_losses, value_losses = [], []
        logs = dict()
        train_start = time.time()

        self.model.train()
        self.critic.train()
        for _, trajs in enumerate(dataloader):
            (critic_loss) = self.train_step_critic(critic_loss_fn, trajs)
            critic_losses.append(critic_loss)

        self.critic.eval()
        for _, trajs in enumerate(dataloader):
            loss, nll, entropy, mse, value_loss = self.train_step_stochastic(loss_fn, value_loss_fn, trajs)
            losses.append(loss)
            nlls.append(nll)
            entropies.append(entropy)
            mses.append(mse)
            value_losses.append(value_loss)

        logs["time/training"] = time.time() - train_start
        logs["training/train_loss_mean"] = np.mean(losses)
        logs["training/train_loss_std"] = np.std(losses)
        logs["training/nll"] = nlls[-1]
        logs["training/entropy"] = entropies[-1]
        logs["training/mse"] = mses[-1]
        logs["training/train_critic_loss_mean"] = np.mean(critic_losses)
        logs["training/train_critic_loss_std"] = np.std(critic_losses)
        logs["training/train_value_loss_mean"] = np.mean(value_losses)
        logs["training/train_value_loss_std"] = np.std(value_losses)
        logs["training/temp_value"] = self.model.temperature().detach().cpu().item()

        return logs

    def train_step_critic(self, critic_loss_fn, trajs):
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            rtg_discounted,
            timesteps,
            ordering,
            padding_mask,
        ) = trajs

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        rtg_discounted = rtg_discounted.to(self.device)
        timesteps = timesteps.to(self.device)
        ordering = ordering.to(self.device)
        padding_mask = padding_mask.to(self.device)

        rtg_discounted_target = torch.clone(rtg_discounted)

        _, _, return_preds = self.critic.forward(
            states,
            actions,
            rewards,
            rtg_discounted[:, :-1],
            timesteps,
            ordering,
            padding_mask=padding_mask,
        )

        critic_loss, critic_mse, critic_nll, rtg, rtg_hat_mean, rtg_hat_std = critic_loss_fn(
            return_preds,
            rtg_discounted_target[:, :-1],
            padding_mask,
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.25)
        self.critic_optimizer.step()

        return (
            critic_loss.detach().cpu().item(),
        )

    def train_step_stochastic(self, loss_fn, value_loss_fn, trajs):
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            rtg_discounted,
            timesteps,
            ordering,
            padding_mask,
        ) = trajs

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        rtg = rtg.to(self.device)
        rtg_discounted = rtg_discounted.to(self.device)
        timesteps = timesteps.to(self.device)
        ordering = ordering.to(self.device)
        padding_mask = padding_mask.to(self.device)

        action_target = torch.clone(actions)
        rtg_discounted_target = torch.clone(rtg_discounted)

        _, action_preds, _ = self.model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            ordering,
            padding_mask=padding_mask,
        )

        loss, nll, entropy, mse = loss_fn(
            action_preds,  # a_hat_dist
            action_target,
            padding_mask,
            self.model.temperature().detach(),  # no gradient taken here
        )

        _, _, return_preds = self.critic.forward(
            states,
            action_preds.mean,
            rewards,
            rtg_discounted[:, :-1],
            timesteps,
            ordering,
            padding_mask=padding_mask,
        )

        value_loss, value_mse, value_nll, rtg, rtg_hat_mean, rtg_hat_std = value_loss_fn(
            return_preds,
            rtg_discounted_target[:, :-1],
            padding_mask,
        )

        self.optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        ((loss + value_loss * self.value_coef) / np.abs(1 + self.value_coef)).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        self.log_temperature_optimizer.zero_grad()
        temperature_loss = (
            self.model.temperature() * (entropy - self.model.target_entropy).detach()
        )
        temperature_loss.backward()
        self.log_temperature_optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return (
            loss.detach().cpu().item(),
            nll.detach().cpu().item(),
            entropy.detach().cpu().item(),
            mse.detach().cpu().item(),
            value_loss.detach().cpu().item(),
        )


