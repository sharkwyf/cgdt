"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.

Part of the code was adapted from the following: 
* https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py
* https://github.com/denisyarats/pytorch_sac/blob/master/agent/actor.py

Both are licensed under the MIT License.
"""

import torch
import torch.nn as nn

import transformers

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model
import math
import numpy as np
import torch.nn.functional as F
from torch import distributions as pyd


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    """
    Squashed Normal Distribution(s)

    If loc/std is of size (batch_size, sequence length, d),
    this returns batch_size * sequence length * d
    independent squashed univariate normal distributions.
    """

    def __init__(self, loc, std):
        self.loc = loc
        self.std = std
        self.base_dist = pyd.Normal(loc, std)

        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self, N=1):
        # sample from the distribution and then compute
        # the empirical entropy:
        x = self.rsample((N,))
        log_p = self.log_prob(x)

        # log_p: (batch_size, context_len, action_dim),
        return -log_p.mean(axis=0).sum(axis=2)

    def log_likelihood(self, x):
        # log_prob(x): (batch_size, context_len, action_dim)
        # sum up along the action dimensions
        # Return tensor shape: (batch_size, context_len)
        return self.log_prob(x).sum(axis=2)


class Gaussian:
    """ Represents a gaussian distribution """
    # TODO: implement a dict conversion function
    def __init__(self, mu, log_sigma=None):
        """

        :param mu:
        :param log_sigma: If none, mu is divided into two chunks, mu and log_sigma
        """
        if log_sigma is None:
            if not isinstance(mu, torch.Tensor):
                import pdb; pdb.set_trace()
            mu, log_sigma = torch.chunk(mu, 2, -1)

        self.mu = mu
        self.log_sigma = torch.clamp(log_sigma, min=-10, max=2) if isinstance(log_sigma, torch.Tensor) else \
                            np.clip(log_sigma, a_min=-10, a_max=2)
        self._sigma = None

    @staticmethod
    def ten2ar(tensor):
        if isinstance(tensor, np.ndarray):
            return tensor
        elif torch.is_tensor(tensor):
            return tensor.detach().cpu().numpy()
        elif np.isscalar(tensor):
            return tensor
        elif hasattr(tensor, 'to_numpy'):
            return tensor.to_numpy()
        else:
            import pdb; pdb.set_trace()
            raise ValueError('input to ten2ar cannot be converted to numpy array')

    def sample(self):
        return self.mu + self.sigma * torch.randn_like(self.sigma)

    def kl_divergence(self, other):
        """Here self=q and other=p and we compute KL(q, p)"""
        return (other.log_sigma - self.log_sigma) + (self.sigma ** 2 + (self.mu - other.mu) ** 2) \
               / (2 * other.sigma ** 2) - 0.5

    def nll(self, x):
        # Negative log likelihood (probability)
        return -1 * self.log_prob(x)

    def log_prob(self, val):
        """Computes the log-probability of a value under the Gaussian distribution."""
        return -1 * ((val - self.mu) ** 2) / (2 * self.sigma**2) - self.log_sigma - math.log(math.sqrt(2*math.pi))

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.sigma)

    @property
    def sigma(self):
        if self._sigma is None:
            self._sigma = self.log_sigma.exp()
        return self._sigma

    @property
    def shape(self):
        return self.mu.shape

    @staticmethod
    def stack(*argv, dim):
        return Gaussian._combine(torch.stack, *argv, dim=dim)

    @staticmethod
    def cat(*argv, dim):
        return Gaussian._combine(torch.cat, *argv, dim=dim)

    @staticmethod
    def _combine(fcn, *argv, dim):
        mu, log_sigma = [], []
        for g in argv:
            mu.append(g.mu)
            log_sigma.append(g.log_sigma)
        mu = fcn(mu, dim)
        log_sigma = fcn(log_sigma, dim)
        return Gaussian(mu, log_sigma)

    def average(self, dists):
        """Fits single Gaussian to a list of Gaussians."""
        mu_avg = torch.stack([d.mu for d in dists]).sum(0) / len(dists)
        sigma_avg = torch.stack([d.mu ** 2 + d.sigma ** 2 for d in dists]).sum(0) - mu_avg**2
        return type(self)(mu_avg, torch.log(sigma_avg))

    def chunk(self, *args, **kwargs):
        return [type(self)(chunk) for chunk in torch.chunk(self.tensor(), *args, **kwargs)]

    def view(self, shape):
        self.mu = self.mu.view(shape)
        self.log_sigma = self.log_sigma.view(shape)
        self._sigma = self.sigma.view(shape)
        return self

    def __getitem__(self, item):
        return Gaussian(self.mu[item], self.log_sigma[item])

    def tensor(self):
        return torch.cat([self.mu, self.log_sigma], dim=-1)

    def rsample(self):
        """Identical to self.sample(), to conform with pytorch naming scheme."""
        return self.sample()

    def detach(self):
        """Detaches internal variables. Returns detached Gaussian."""
        return type(self)(self.mu.detach(), self.log_sigma.detach())

    def to_numpy(self):
        """Convert internal variables to numpy arrays."""
        return type(self)(self.ten2ar(self.mu), self.ten2ar(self.log_sigma))


class MultivariateGaussian(Gaussian):
    def log_prob(self, val):
        return super().log_prob(val).sum(-1)

    @staticmethod
    def stack(*argv, dim):
        return MultivariateGaussian(Gaussian.stack(*argv, dim=dim).tensor())

    @staticmethod
    def cat(*argv, dim):
        return MultivariateGaussian(Gaussian.cat(*argv, dim=dim).tensor())


class DiagGaussian(nn.Module):
    def __init__(self, hidden_dim, z_dim):
        super().__init__()

        self.mu = nn.Linear(hidden_dim, z_dim)
        self.log_std = nn.Linear(hidden_dim, z_dim)

        def weight_init(m):
            """Custom weight init for Conv2D and Linear layers."""
            if isinstance(m, torch.nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if hasattr(m.bias, "data"):
                    m.bias.data.fill_(0.0)

        self.apply(weight_init)

    def forward(self, x):
        mu = self.mu(x)
        log_std = self.log_std(x)

        return MultivariateGaussian(mu, log_std)


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, hidden_dim, act_dim, log_std_bounds=[-5.0, 2.0]):
        super().__init__()

        self.mu = torch.nn.Linear(hidden_dim, act_dim)
        self.log_std = torch.nn.Linear(hidden_dim, act_dim)
        self.log_std_bounds = log_std_bounds

        def weight_init(m):
            """Custom weight init for Conv2D and Linear layers."""
            if isinstance(m, torch.nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if hasattr(m.bias, "data"):
                    m.bias.data.fill_(0.0)

        self.apply(weight_init)

    def forward(self, obs):
        mu, log_std = self.mu(obs), self.log_std(obs)
        log_std = torch.tanh(log_std)
        # log_std is the output of tanh so it will be between [-1, 1]
        # map it to be between [log_std_min, log_std_max]
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0)
        std = log_std.exp()
        return SquashedNormal(mu, std)


class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_size,
        action_range,
        ordering=0,
        max_length=None,
        eval_context_length=None,
        max_ep_len=4096,
        action_tanh=True,
        stochastic_policy=False,
        discrete_action=False,
        init_temperature=0.1,
        target_entropy=None,
        **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        if ordering:
            self.embed_ordering = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_return = torch.nn.Linear(hidden_size, 1)
        if stochastic_policy:
            if discrete_action:
                self.predict_action = nn.Sequential(
                    *(
                        [nn.Linear(hidden_size, self.act_dim)]
                        + ([nn.Tanh()] if action_tanh else [])
                    )
                )
            else:
                self.predict_action = DiagGaussianActor(hidden_size, self.act_dim)
        else:
            self.predict_action = nn.Sequential(
                *(
                    [nn.Linear(hidden_size, self.act_dim)]
                    + ([nn.Tanh()] if action_tanh else [])
                )
            )
        self.stochastic_policy = stochastic_policy
        self.discrete_action = discrete_action
        self.eval_context_length = eval_context_length
        self.ordering = ordering
        self.action_range = action_range

        if stochastic_policy:
            self.log_temperature = torch.tensor(np.log(init_temperature))
            self.log_temperature.requires_grad = True
            self.target_entropy = target_entropy

    def temperature(self):
        if self.stochastic_policy:
            return self.log_temperature.exp()
        else:
            return None

    def forward(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        ordering,
        padding_mask=None,
    ):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if padding_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            padding_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)

        if self.ordering:
            order_embeddings = self.embed_ordering(timesteps)
        else:
            order_embeddings = 0.0

        state_embeddings = state_embeddings + order_embeddings
        action_embeddings = action_embeddings + order_embeddings
        returns_embeddings = returns_embeddings + order_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            )
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_padding_mask = (
            torch.stack((padding_mask, padding_mask, padding_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_padding_mask,
        )
        x = transformer_outputs["last_hidden_state"]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        # predict next return given state and action
        return_preds = self.predict_return(x[:, 2])
        # predict next state given state and action
        state_preds = self.predict_state(x[:, 2])
        # predict next action given state
        action_preds = self.predict_action(x[:, 1])

        if self.discrete_action:
            action_preds = action_preds.softmax(dim=-1)

        return state_preds, action_preds, return_preds

    def get_predictions(
        self, states, actions, rewards, returns_to_go, timesteps, num_envs=1, **kwargs
    ):
        # we don't care about the past rewards in this model
        # tensor shape: batch_size, seq_length, variable_dim
        states = states.reshape(num_envs, -1, self.state_dim)
        actions = actions.reshape(num_envs, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(num_envs, -1, 1)

        # tensor shape: batch_size, seq_length
        timesteps = timesteps.reshape(num_envs, -1)

        # max_length is the DT context length (should be input length of the subsequence)
        # eval_context_length is the how long you want to use the history for your prediction
        if self.max_length is not None:
            states = states[:, -self.eval_context_length :]
            actions = actions[:, -self.eval_context_length :]
            returns_to_go = returns_to_go[:, -self.eval_context_length :]
            timesteps = timesteps[:, -self.eval_context_length :]

            ordering = torch.tile(
                torch.arange(timesteps.shape[1], device=states.device),
                (num_envs, 1),
            )
            # pad all tokens to sequence length
            padding_mask = torch.cat(
                [
                    torch.zeros(self.max_length - states.shape[1]),
                    torch.ones(states.shape[1]),
                ]
            )
            padding_mask = padding_mask.to(
                dtype=torch.long, device=states.device
            ).reshape(1, -1)
            padding_mask = padding_mask.repeat((num_envs, 1))

            states = torch.cat(
                [
                    torch.zeros(
                        (
                            states.shape[0],
                            self.max_length - states.shape[1],
                            self.state_dim,
                        ),
                        device=states.device,
                    ),
                    states,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            actions = torch.cat(
                [
                    torch.zeros(
                        (
                            actions.shape[0],
                            self.max_length - actions.shape[1],
                            self.act_dim,
                        ),
                        device=actions.device,
                    ),
                    actions,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [
                    torch.zeros(
                        (
                            returns_to_go.shape[0],
                            self.max_length - returns_to_go.shape[1],
                            1,
                        ),
                        device=returns_to_go.device,
                    ),
                    returns_to_go,
                ],
                dim=1,
            ).to(dtype=torch.float32)

            timesteps = torch.cat(
                [
                    torch.zeros(
                        (timesteps.shape[0], self.max_length - timesteps.shape[1]),
                        device=timesteps.device,
                    ),
                    timesteps,
                ],
                dim=1,
            ).to(dtype=torch.long)

            ordering = torch.cat(
                [
                    torch.zeros(
                        (ordering.shape[0], self.max_length - ordering.shape[1]),
                        device=ordering.device,
                    ),
                    ordering,
                ],
                dim=1,
            ).to(dtype=torch.long)
        else:
            padding_mask = None

        state_preds, action_preds, return_preds = self.forward(
            states,
            actions,
            None,
            returns_to_go,
            timesteps,
            ordering,
            padding_mask=padding_mask,
            **kwargs
        )
        if self.stochastic_policy:
            return state_preds[:, -1], action_preds, return_preds[:, -1]
        else:
            return (
                state_preds[:, -1],
                self.clamp_action(action_preds[:, -1]),
                return_preds[:, -1],
            )

    def clamp_action(self, action):
        return action.clamp(*self.action_range)


class ImitationTransformer(TrajectoryModel):

    """
    This model uses GPT to model (state_1, action_1, state_2, ...)
    """

    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_size,
        action_range,
        ordering=0,
        max_length=None,
        eval_context_length=None,
        max_ep_len=4096,
        action_tanh=True,
        stochastic_policy=False,
        discrete_action=False,
        init_temperature=0.1,
        target_entropy=None,
        **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        if ordering:
            self.embed_ordering = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        # self.predict_return = torch.nn.Linear(hidden_size, 1)
        self.predict_return = DiagGaussian(hidden_size, 1)
        if stochastic_policy:
            self.predict_action = DiagGaussianActor(hidden_size, self.act_dim)
        else:
            self.predict_action = nn.Sequential(
                *(
                    [nn.Linear(hidden_size, self.act_dim)]
                    + ([nn.Tanh()] if action_tanh else [])
                )
            )
        self.stochastic_policy = stochastic_policy
        self.discrete_action = discrete_action
        self.eval_context_length = eval_context_length
        self.ordering = ordering
        self.action_range = action_range

        if stochastic_policy:
            self.log_temperature = torch.tensor(np.log(init_temperature))
            self.log_temperature.requires_grad = True
            self.target_entropy = target_entropy

    def temperature(self):
        if self.stochastic_policy:
            return self.log_temperature.exp()
        else:
            return None

    def forward(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        ordering,
        padding_mask=None,
    ):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if padding_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            padding_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)

        if self.ordering:
            order_embeddings = self.embed_ordering(timesteps)
        else:
            order_embeddings = 0.0

        state_embeddings = state_embeddings + order_embeddings
        action_embeddings = action_embeddings + order_embeddings
        returns_embeddings = returns_embeddings + order_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack(
                (state_embeddings, action_embeddings), dim=1
            )
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 2 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_padding_mask = (
            torch.stack((padding_mask, padding_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 2 * seq_length)
        )

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_padding_mask,
        )
        x = transformer_outputs["last_hidden_state"]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        # predict next return given state and action
        return_preds = self.predict_return(x[:, 1])
        # predict next state given state and action
        state_preds = self.predict_state(x[:, 1])
        # predict next action given state
        action_preds = self.predict_action(x[:, 0])

        return state_preds, action_preds, return_preds

    def get_predictions(
        self, states, actions, rewards, returns_to_go, timesteps, num_envs=1, **kwargs
    ):
        # we don't care about the past rewards in this model
        # tensor shape: batch_size, seq_length, variable_dim
        states = states.reshape(num_envs, -1, self.state_dim)
        actions = actions.reshape(num_envs, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(num_envs, -1, 1)

        # tensor shape: batch_size, seq_length
        timesteps = timesteps.reshape(num_envs, -1)

        # max_length is the DT context length (should be input length of the subsequence)
        # eval_context_length is the how long you want to use the history for your prediction
        if self.max_length is not None:
            states = states[:, -self.eval_context_length :]
            actions = actions[:, -self.eval_context_length :]
            returns_to_go = returns_to_go[:, -self.eval_context_length :]
            timesteps = timesteps[:, -self.eval_context_length :]

            ordering = torch.tile(
                torch.arange(timesteps.shape[1], device=states.device),
                (num_envs, 1),
            )
            # pad all tokens to sequence length
            padding_mask = torch.cat(
                [
                    torch.zeros(self.max_length - states.shape[1]),
                    torch.ones(states.shape[1]),
                ]
            )
            padding_mask = padding_mask.to(
                dtype=torch.long, device=states.device
            ).reshape(1, -1)
            padding_mask = padding_mask.repeat((num_envs, 1))

            states = torch.cat(
                [
                    torch.zeros(
                        (
                            states.shape[0],
                            self.max_length - states.shape[1],
                            self.state_dim,
                        ),
                        device=states.device,
                    ),
                    states,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            actions = torch.cat(
                [
                    torch.zeros(
                        (
                            actions.shape[0],
                            self.max_length - actions.shape[1],
                            self.act_dim,
                        ),
                        device=actions.device,
                    ),
                    actions,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [
                    torch.zeros(
                        (
                            returns_to_go.shape[0],
                            self.max_length - returns_to_go.shape[1],
                            1,
                        ),
                        device=returns_to_go.device,
                    ),
                    returns_to_go,
                ],
                dim=1,
            ).to(dtype=torch.float32)

            timesteps = torch.cat(
                [
                    torch.zeros(
                        (timesteps.shape[0], self.max_length - timesteps.shape[1]),
                        device=timesteps.device,
                    ),
                    timesteps,
                ],
                dim=1,
            ).to(dtype=torch.long)

            ordering = torch.cat(
                [
                    torch.zeros(
                        (ordering.shape[0], self.max_length - ordering.shape[1]),
                        device=ordering.device,
                    ),
                    ordering,
                ],
                dim=1,
            ).to(dtype=torch.long)
        else:
            padding_mask = None

        state_preds, action_preds, return_preds = self.forward(
            states,
            actions,
            None,
            returns_to_go,
            timesteps,
            ordering,
            padding_mask=padding_mask,
            **kwargs
        )
        if self.stochastic_policy:
            return state_preds[:, -1], action_preds, return_preds[:, -1]
        else:
            return (
                state_preds[:, -1],
                self.clamp_action(action_preds[:, -1]),
                return_preds[:, -1],
            )

    def clamp_action(self, action):
        return action.clamp(*self.action_range)