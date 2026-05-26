"""LTX-2 flow-matching scheduler.

Mirrors the training-time σ sampling distribution used by the LTX-2/2.3
reference trainer (`ltx_trainer.timestep_samplers.ShiftedLogitNormalTimestepSampler`):

  mu = lerp(min_shift, max_shift, seq_length, [min_tokens, max_tokens])
  normal ~ N(mu, std)
  logit_normal = sigmoid(normal)
  zero_terminal = (logit_normal - p005) / (p999 - p005)        # stretch to [0,1]
  stretched = where(zero_terminal >= eps, zero_terminal, 2*eps - zero_terminal)
  stretched = clamp(stretched, 0, 1)
  uniform = (1 - eps) * U(0,1) + eps
  sigma = where(rand > uniform_prob, stretched, uniform)
  timestep = sigma * num_train_timesteps

The non-sampling methods (add_noise, training_target, build_inference_schedule,
step) keep the same contract as `WanContinuousFlowMatchScheduler` so the rest
of the model code is unchanged.

Reference: third_party/ltx-2/packages/ltx-trainer/src/ltx_trainer/timestep_samplers.py
"""

from __future__ import annotations

import torch


class LTX2FlowMatchScheduler:
    """Stretched, shifted logit-normal flow-matching σ sampler.

    Shift `mu` is a linear function of sequence length, matching the LTX-2.3
    paper / reference trainer. Wider sequences get larger mu (mean sigma).
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        min_shift: float = 0.95,
        max_shift: float = 2.05,
        min_tokens: int = 1024,
        max_tokens: int = 4096,
        std: float = 1.0,
        eps: float = 1e-3,
        uniform_prob: float = 0.1,
        infer_shift: float = 2.05,
    ) -> None:
        if num_train_timesteps <= 0:
            raise ValueError(f"`num_train_timesteps` must be positive, got {num_train_timesteps}")
        if max_tokens <= min_tokens:
            raise ValueError(
                f"`max_tokens` must be > `min_tokens` (got {max_tokens} <= {min_tokens})"
            )
        if std <= 0:
            raise ValueError(f"`std` must be positive, got {std}")
        if not 0.0 <= uniform_prob <= 1.0:
            raise ValueError(f"`uniform_prob` must be in [0, 1], got {uniform_prob}")
        if not 0.0 < eps < 0.5:
            raise ValueError(f"`eps` must be in (0, 0.5), got {eps}")
        self.num_train_timesteps = int(num_train_timesteps)
        self.min_shift = float(min_shift)
        self.max_shift = float(max_shift)
        self.min_tokens = int(min_tokens)
        self.max_tokens = int(max_tokens)
        self.std = float(std)
        self.eps = float(eps)
        self.uniform_prob = float(uniform_prob)
        self.infer_shift = float(infer_shift)
        # ±~3 std percentiles (in normal space), scaled by std.
        self._normal_999 = 3.0902 * self.std
        self._normal_005 = -2.5758 * self.std

    def _mu_for_seq_length(self, seq_length: int) -> float:
        m = (self.max_shift - self.min_shift) / (self.max_tokens - self.min_tokens)
        b = self.min_shift - m * self.min_tokens
        return m * float(seq_length) + b

    def sample_training_t(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        seq_length: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}")
        if seq_length is None:
            raise ValueError(
                "LTX2FlowMatchScheduler.sample_training_t requires `seq_length` "
                "(shift varies with token count). Pass it from the caller."
            )

        mu = self._mu_for_seq_length(int(seq_length))

        normal = torch.randn((batch_size,), device=device, dtype=torch.float32) * self.std + mu
        logit_normal = torch.sigmoid(normal)

        # Stretch to span [0, 1] via percentile bounds of the same shifted normal.
        p999 = torch.sigmoid(torch.tensor(mu + self._normal_999, device=device, dtype=torch.float32))
        p005 = torch.sigmoid(torch.tensor(mu + self._normal_005, device=device, dtype=torch.float32))
        zero_terminal = (logit_normal - p005) / (p999 - p005)

        # Reflect tiny values around `eps` for numerical stability near zero.
        stretched = torch.where(
            zero_terminal >= self.eps,
            zero_terminal,
            2.0 * self.eps - zero_terminal,
        )
        stretched = torch.clamp(stretched, 0.0, 1.0)

        # Mix in `uniform_prob` fraction of uniform samples to prevent distribution
        # collapse at extreme shifts.
        uniform = (1.0 - self.eps) * torch.rand((batch_size,), device=device, dtype=torch.float32) + self.eps
        prob = torch.rand((batch_size,), device=device, dtype=torch.float32)
        sigma = torch.where(prob > self.uniform_prob, stretched, uniform)

        timestep = sigma * float(self.num_train_timesteps)
        return timestep.to(dtype=dtype)

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        # LTX reference does not apply a training-weight curve on top of the
        # sampling distribution. Return uniform 1's to match the loss-averaging
        # contract expected by trainer code without re-weighting.
        return torch.ones_like(timestep, dtype=torch.float32)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        sigma = (timestep / float(self.num_train_timesteps)).to(
            original_samples.device, dtype=original_samples.dtype
        )
        if sigma.ndim == 0:
            return (1 - sigma) * original_samples + sigma * noise
        sigma = sigma.view(-1, *([1] * (original_samples.ndim - 1)))
        return (1 - sigma) * original_samples + sigma * noise

    @staticmethod
    def training_target(
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        del timestep
        return noise - sample

    def build_inference_schedule(
        self,
        num_inference_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        shift_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}")
        shift = self.infer_shift if shift_override is None else float(shift_override)
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")

        # Use the Möbius shift over a 1→0 linspace, matching the Wan inference
        # schedule (same flow-matching ODE, parameterization choice for the σ
        # spacing is independent of the training sampler). The reference LTX
        # pipeline uses a more elaborate `LTX2Scheduler`; if a future change
        # demands the full schedule, swap this method only.
        u_steps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.float32)
        sigma_steps = shift * u_steps / (1.0 + (shift - 1.0) * u_steps)
        timesteps = sigma_steps[:-1] * float(self.num_train_timesteps)
        deltas = sigma_steps[1:] - sigma_steps[:-1]
        return timesteps.to(dtype=dtype), deltas.to(dtype=dtype)

    @staticmethod
    def step(model_output: torch.Tensor, delta: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        delta = delta.to(sample.device, dtype=sample.dtype)
        if delta.ndim == 0:
            return sample + model_output * delta
        delta = delta.view(-1, *([1] * (sample.ndim - 1)))
        return sample + model_output * delta
