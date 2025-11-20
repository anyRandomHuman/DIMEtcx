import os

import jax
import flax
import numpy as np
import jax.numpy as jnp

from gymnasium import spaces
from functools import partial

from diffusion.diffusion_policy import DiffPol
from flax.training.train_state import TrainState
from stable_baselines3.common.noise import ActionNoise
from diffusion.dime import DIME
from common.type_aliases import ReplayBufferSamplesMTNP, RLTrainState
from typing import Any, ClassVar, Dict, Optional, Tuple, Type, Union
from stable_baselines3.common.type_aliases import GymEnv
from common.buffers import MTReplayBuffer
from common.normalizer import RewardNormalizer


class MTDIME(DIME):
    policy_aliases: ClassVar[Dict[str, Type[DiffPol]]] = {  # type: ignore[assignment]
        "MlpPolicy": DiffPol,
        # Minimal dict support using flatten()
        "MultiInputPolicy": DiffPol,
    }

    policy: DiffPol
    action_space: spaces.Box  # type: ignore[assignment]

    def __init__(self,
                 policy,
                 env: Union[GymEnv, str],
                 model_save_path: str,
                 save_every_n_steps: int,
                 cfg,
                 train_freq: Union[int, Tuple[int, str]] = 1,
                 action_noise: Optional[ActionNoise] = None,
                 replay_buffer_class: Optional[Type[MTReplayBuffer]] = None,
                 replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
                 use_sde: bool = False,
                 sde_sample_freq: int = -1,
                 use_sde_at_warmup: bool = False,
                 tensorboard_log: Optional[str] = None,
                 verbose: int = 0,
                 _init_setup_model: bool = True,
                 stats_window_size: int = 100,
                 ) -> None:
        self.n_tasks = cfg.n_tasks
        super().__init__(
            policy=policy,
            env=env,
            model_save_path=model_save_path,
            save_every_n_steps=save_every_n_steps,
            cfg=cfg,
            train_freq=train_freq,
            action_noise=action_noise,
            replay_buffer_class=MTReplayBuffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            _init_setup_model=_init_setup_model,
            stats_window_size=stats_window_size,
        )
        self.normalize_reward = cfg.alg.normalize_reward
        self.normalizer = RewardNormalizer(env.num_envs, self.target_entropy, discount=cfg.alg.gamma, v_max=cfg.alg.vmax) if self.normalize_reward else None


    def train(self, batch_size, gradient_steps):
        # Sample all at once for efficiency (so we can jit the for loop)
        data = self.replay_buffer.sample(batch_size * gradient_steps, env=self._vec_normalize_env)
        # Pre-compute the indices where we need to update the actor
        # This is a hack in order to jit the train loop
        # It will compile once per value of policy_delay_indices
        policy_delay_indices = {i: True for i in range(gradient_steps) if
                                ((self._n_updates + i + 1) % self.policy_delay) == 0}
        policy_delay_indices = flax.core.FrozenDict(policy_delay_indices)

        if isinstance(data.observations, dict):
            keys = list(self.observation_space.keys())
            obs = np.concatenate([data.observations[key].numpy() for key in keys], axis=1)
            next_obs = np.concatenate([data.next_observations[key].numpy() for key in keys], axis=1)
        else:
            obs = data.observations.numpy()
            next_obs = data.next_observations.numpy()

        # Convert to numpy
        data = ReplayBufferSamplesMTNP(
            obs,
            data.actions.numpy(),
            next_obs,
            data.dones.numpy().flatten(),
            data.rewards.numpy().flatten(),
            data.task_ids.numpy().flatten()
        )

        if self.normalize_reward:
            data = self.normalizer.normalize(data, temperature=self.current_entropy_coeff())

        (
            self.policy.qf_state,
            self.policy.actor_state,
            self.policy.target_actor_state,
            self.ent_coef_state,
            self.key,
            log_metrics,
        ) = self._train(
            self.crossq_style,
            self.use_bnstats_from_live_net,
            self.gamma,
            self.tau,
            self.policy_tau,
            self.target_entropy,
            gradient_steps,
            data,
            policy_delay_indices,
            self.policy.qf_state,
            self.policy.actor_state,
            self.policy.target_actor_state,
            self.ent_coef_state,
            self.key,
            self.num_timesteps,
            self.policy_q_reduce_fn,
            self.policy.sampler,
            self.policy.target_sampler,
            self.cfg.alg.critic.v_min,
            self.cfg.alg.critic.v_max,
            self.cfg.alg.critic.entr_coeff,
            self.cfg.alg.critic.n_atoms
        )
        self._n_updates += gradient_steps

        if self.model_save_path is not None:
            if (self.num_timesteps % self.save_every_n_steps == 0) or (self.num_timesteps == (self.learning_starts+1)):
                self._save_model()

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        for k, v in log_metrics.items():
            try:
                log_val = v.item()
            except:
                log_val = v
            self.logger.record(f"train/{k}", log_val)

    @classmethod
    @partial(jax.jit,
             static_argnames=["cls", "crossq_style", "use_bnstats_from_live_net", "gradient_steps", "q_reduce_fn",
                              "sampler", "target_sampler", "v_min", "v_max", "num_atoms", "entr_coeff"])
    def _train(
            cls,
            crossq_style: bool,
            use_bnstats_from_live_net: bool,
            gamma: float,
            tau: float,
            policy_tau: float,
            target_entropy: np.ndarray,
            gradient_steps: int,
            data: ReplayBufferSamplesMTNP,
            policy_delay_indices: flax.core.FrozenDict,
            qf_state: RLTrainState,
            actor_state: TrainState,
            target_actor_state: TrainState,
            ent_coef_state: TrainState,
            key,
            n_env_interacts,
            q_reduce_fn,
            sampler,
            target_sampler,
            v_min,
            v_max,
            entr_coeff,
            num_atoms
    ):
        actor_loss_value = jnp.array(0)
        actor_metrics = [{}]
        for i in range(gradient_steps):

            def slice(x, step=i):
                assert x.shape[0] % gradient_steps == 0
                batch_size = x.shape[0] // gradient_steps
                return x[batch_size * step: batch_size * (step + 1)]

            z_atoms = jnp.linspace(v_min,  v_max, num_atoms)

            (
                qf_state,
                log_metrics_critic,
                key,
            ) = cls.update_critic(
                crossq_style,
                use_bnstats_from_live_net,
                gamma,
                target_actor_state,
                qf_state,
                ent_coef_state,
                slice(data.observations),
                slice(data.actions),
                slice(data.next_observations),
                slice(data.rewards),
                slice(data.dones),
                n_env_interacts,
                num_atoms,
                z_atoms,
                v_min,
                v_max,
                entr_coeff,
                key,
                target_sampler
            )
            qf_state = MTDIME.soft_update(tau, qf_state)
            target_actor_state = target_actor_state
            # hack to be able to jit (n_updates % policy_delay == 0)
            # a = False
            if i in policy_delay_indices:  # and a:
                (actor_state, qf_state, actor_loss_value, key, actor_metrics) = cls.update_actor(
                    actor_state,
                    qf_state,
                    ent_coef_state,
                    slice(data.observations),
                    n_env_interacts,
                    key,
                    z_atoms,
                    sampler,
                    q_reduce_fn,
                )
                ent_coef_state, _ = MTDIME.update_temperature(target_entropy, ent_coef_state,
                                                              actor_metrics[0]['run_costs'])

                target_actor_state = MTDIME.soft_update_target_actor(policy_tau, actor_state, target_actor_state)
        log_metrics = {'actor_loss': actor_loss_value, **actor_metrics[0], **log_metrics_critic}
        return qf_state, actor_state, target_actor_state, ent_coef_state, key, log_metrics

    def _setup_model(self, reset=False) -> None:
        if not reset:
            self._setup_lr_schedule()
            # By default qf_learning_rate = pi_learning_rate
            self.qf_learning_rate = self.qf_learning_rate or self.lr_schedule(1)
            self.set_random_seed(self.seed)
            # Make a local copy as we should not pickle
            # the environment when using HerReplayBuffer
            replay_buffer_kwargs = self.replay_buffer_kwargs.copy()

            self.replay_buffer = MTReplayBuffer(  # type: ignore[misc]
                self.buffer_size,
                self.observation_space,
                self.action_space,
                num_tasks=self.n_tasks,
                device="cpu",  # force cpu device to easy torch -> numpy conversion
                n_envs=self.n_envs,
                optimize_memory_usage=self.optimize_memory_usage,
                **replay_buffer_kwargs,
            )
            # Convert train freq parameter to TrainFreq object
            self._convert_train_freq()

        if not hasattr(self, "policy") or self.policy is None or reset:
            super()._setup_model(reset=True)

    def collect_rollouts(
        self,
        env,
        callback,
        train_freq,
        replay_buffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ):
        r = super().collect_rollouts(env, callback, train_freq,replay_buffer, action_noise,learning_starts,log_interval)
        if self.normalize_reward:
            reward = self.replay_buffer.rewards[self.replay_buffer.pos]
            dones = self.replay_buffer.dones[self.replay_buffer.pos]
            self.normalizer.update(reward, dones)
        return r