import logging
import json
import inspect
import os
import re
import subprocess
from math import ceil
from pathlib import Path
import time

from datetime import timedelta

import numpy as np
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        # Batches each worker pre-loads ahead of demand. The dataset decodes
        # multi-camera video from .mp4 on the fly (variable CPU cost), so a
        # deeper prefetch queue smooths per-step jitter. Default 4 (DataLoader's
        # own default is 2); inert when num_workers == 0.
        self.prefetch_factor = int(cfg.get("prefetch_factor", 4))
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        # Checkpoint rotation knobs (defaulted in train.yaml). state is ~7x
        # bigger than weights for A14B, so we always keep at most 1 state
        # (resume from there), and let weights accumulate long-term anchors.
        self.state_keep_last_n = int(cfg.get("state_keep_last_n", 1))
        self.weights_keep_last_n = int(cfg.get("weights_keep_last_n", 1))
        self.long_term_save_every = int(cfg.get("long_term_save_every", 0))
        # State long-term retention (separate from weights):
        #   keep state at steps where step % state_long_term_save_every == 0
        #   AND step >= state_long_term_start. 0 = disabled (default).
        self.state_long_term_save_every = int(cfg.get("state_long_term_save_every", 0))
        self.state_long_term_start = int(cfg.get("state_long_term_start", 0))
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        # torch.compile the MoT transformer blocks (block-level so it composes
        # with FSDP's per-MoTBlock wrap). Off by default — needs a GPU smoke
        # test to confirm no graph breaks / recompiles on the real shapes.
        self.compile_mot = bool(cfg.get("compile_mot", False))
        # Optimizer: "adamw" (fp32 moments) or "adamw8bit" (bitsandbytes 8-bit
        # moments, ~halves optimizer-state VRAM). 8-bit is effective under FSDP
        # / ZeRO-1 / DDP where the state lives on GPU; under DeepSpeed ZeRO-2
        # with CPU offload DeepSpeed substitutes its own optimizer and this is
        # only a hyperparameter placeholder.
        self.optimizer_type = str(cfg.get("optimizer_type", "adamw")).strip().lower()
        if self.optimizer_type not in {"adamw", "adamw8bit"}:
            raise ValueError(
                f"Unsupported optimizer_type: {cfg.get('optimizer_type')}. "
                "Expected one of: ['adamw', 'adamw8bit']."
            )
        
        self.resume = cfg.resume
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        # NCCL collective timeout. PyTorch's default is 10 min, which is too
        # tight for large FSDP runs: a transient filesystem hiccup or GC pause
        # on any one rank can stall a single small collective (e.g. the
        # grad-norm all-gather) past 10 min, the watchdog fires, and every rank
        # SIGABRTs — losing the run. 30 min absorbs most stalls.
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=30))
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[init_kwargs],
        )

        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)

        ds_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if ds_plugin is not None:
            zero_stage = ds_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown")
        else:
            zero_stage = "n/a (non-deepspeed)"
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            zero_stage,
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model)
        trainable_params = list(self.model.dit.parameters())
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            trainable_params.extend(list(proprio_encoder.parameters()))
        # DeepSpeed Zero-2 with offload_optimizer.device=cpu will route this through
        # DeepSpeedCPUAdam automatically; the constructor here is just the
        # placeholder DeepSpeed inspects for hyperparameters.
        if self.optimizer_type == "adamw8bit":
            try:
                from bitsandbytes.optim import AdamW8bit
            except ImportError as e:
                raise ImportError(
                    "optimizer_type='adamw8bit' requires the `bitsandbytes` package."
                ) from e
            self.optimizer = AdamW8bit(
                trainable_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.95),
            )
            logger.info("Optimizer: bitsandbytes AdamW8bit (8-bit moments).")
        else:
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.95),
            )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self._maybe_compile_mot()
        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)

        # --- bitsandbytes-8bit x FSDP checkpoint incompatibility ----------
        # `accelerator.save_state` routes the optimizer through
        # `FSDP.optim_state_dict`, whose `_convert_all_state_info` machinery
        # assumes every optimizer-state tensor for a parameter has the same
        # numel as the (unsharded) parameter, so it can all-gather / reshard
        # it. bitsandbytes 8-bit optimizers violate that: their per-param
        # state is block-quantized — `state1`/`state2` (uint8) plus
        # `absmax1`/`absmax2` (fp32, ~1/2048 the size) and `qmap` tensors —
        # none of which share the parameter's numel. FSDP cannot reshape
        # them and crashes with a tensor-size mismatch in
        # `_convert_all_state_info`.
        #
        # So when (and only when) we run AdamW8bit under FSDP, we bypass
        # `accelerator.save_state`/`load_state` for the optimizer and persist
        # the rank-local bnb optimizer shard directly via
        # `self.optimizer.state_dict()` (which does NOT trigger the FSDP
        # gather). Model weights still use the FULL_STATE_DICT `.pt` path
        # (`_save_weights_checkpoint`), which works fine. The plain `adamw`
        # path is unaffected and keeps using `accelerator.save_state`.
        from accelerate import DistributedType as _DistributedType
        self._manual_optim_ckpt = (
            self.optimizer_type == "adamw8bit"
            and self.accelerator.distributed_type == _DistributedType.FSDP
        )
        if self._manual_optim_ckpt:
            logger.info(
                "adamw8bit + FSDP detected: checkpointing will use the manual "
                "per-rank optimizer-state path (bypassing FSDP.optim_state_dict)."
            )

        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            worker_init_fn=worker_init_fn,
            # Keep workers alive across epoch boundaries. The training loop
            # re-creates the dataloader iterator every epoch; without this the
            # worker processes are torn down and respawned each time — and each
            # respawn re-loads the per-worker CPU text encoder. Only valid when
            # num_workers > 0.
            persistent_workers=self.num_workers > 0,
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored under ZeRO2.")

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss — route through self.model(sample) (FSDP-wrapped __call__)
        #    so the root FSDP pre-forward hook all-gathers params (VAE, etc.) for us.
        #    FastWAM.forward delegates to training_loss, so the call shape is identical.
        with self.accelerator.autocast():
            val_loss, _ = self.model(sample)
            val_loss = val_loss.float().item()

        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        # model.infer / model._encode_video_latents / model._decode_latents are
        # method calls (not __call__), so under FSDP we must explicitly gather
        # the root FSDP's params (VAE / proprio_encoder / anything not matched
        # by transformer_layer_cls_to_wrap). MoTBlocks are their own FSDP units
        # and will gather/release on demand inside their per-call forwards —
        # recurse=False keeps their shards intact, avoiding the ~28 GB
        # inflation that recurse=True would cause.
        from accelerate import DistributedType
        from contextlib import nullcontext
        def _fsdp_root_summon():
            if self.accelerator.distributed_type == DistributedType.FSDP:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                return FSDP.summon_full_params(self.model, recurse=False, writeback=False)
            return nullcontext()

        # Eval mp4 reflects the model's autonomous policy behavior:
        # 1) Run action expert solo (no gt) to obtain pred_action.
        # 2) Condition video generation on pred_action.
        # This makes the saved mp4 + downstream PSNR/SSIM measure the end-to-end
        # rollout the model would actually produce, rather than "video given the
        # gt action sequence" (which decouples video quality from policy quality
        # but hides policy failures). action_l1/action_l2 still come from the
        # same pred_action vs gt comparison and remain meaningful.
        with _fsdp_root_summon(), self.accelerator.autocast():
            _pred_only = model.infer_action(
                prompt=infer_kwargs.get("prompt"),
                input_image=input_image,
                action_horizon=sample['action_horizon'],
                context=infer_kwargs.get("context"),
                context_mask=infer_kwargs.get("context_mask"),
                proprio=proprio,
                num_inference_steps=self.eval_num_inference_steps,
                seed=42,
            )
        infer_kwargs["action"] = _pred_only["action"].to(
            device=model.device, dtype=model.torch_dtype
        )

        with _fsdp_root_summon(), self.accelerator.autocast():
            pred = model.infer(
                **infer_kwargs,
            )

        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
            
            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        # model._encode_video_latents / ._decode_latents are direct method calls
        # that touch VAE params; reuse the same FSDP gather + autocast contexts.
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        with _fsdp_root_summon(), self.accelerator.autocast():
            vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
            vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)

        # Free intermediate tensors created during inference (KV cache list,
        # action denoising buffers, decode output). Without this, the next
        # `save_state` collective allocates its plan-exchange tensors against
        # a fragmented allocator and trips NCCL OOM on the dcp reduce_scatter.
        del local_metrics, gathered_metrics, mean_metrics
        torch.cuda.empty_cache()
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        # state_dict() is a collective under FSDP and every rank must call it
        # so the all-gather can complete. The FSDP.state_dict_type context is
        # set on the root model so it propagates to every nested FSDP unit
        # (every MoTBlock); inside the context, submodule.state_dict() works
        # as a coordinated FULL_STATE_DICT gather (rank 0 receives the full
        # unsharded weights, other ranks get empty/sharded dicts but must
        # participate). Only rank 0 writes to disk.
        from accelerate import DistributedType

        inner = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")

        if self.accelerator.distributed_type == DistributedType.FSDP:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                StateDictType,
                FullStateDictConfig,
            )
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                mot_state = inner.mot.state_dict()
                proprio_state = (
                    inner.proprio_encoder.state_dict()
                    if inner.proprio_encoder is not None
                    else None
                )
        else:
            # DeepSpeed Zero-2 / DDP / single-GPU: each rank already holds the
            # full param set, so state_dict() needs no collective.
            mot_state = inner.mot.state_dict()
            proprio_state = (
                inner.proprio_encoder.state_dict()
                if inner.proprio_encoder is not None
                else None
            )

        if self.accelerator.is_main_process:
            payload = {
                "mot": mot_state,
                "step": int(self.global_step),
                "torch_dtype": str(inner.torch_dtype),
            }
            if proprio_state is not None:
                payload["proprio_encoder"] = proprio_state
            torch.save(payload, ckpt_path)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        # For the manual adamw8bit+FSDP path the optimizer state is sharded
        # per-rank, so the checkpoint is only loadable at the same world size.
        # Record it (and a marker) so `load_training_state` can detect the
        # manual layout and validate the world size on resume.
        if self._manual_optim_ckpt:
            payload["manual_optim_ckpt"] = True
            payload["world_size"] = int(self.accelerator.num_processes)
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    @staticmethod
    def _step_from_name(name: str) -> int | None:
        m = re.match(r"step_(\d+)", name)
        return int(m.group(1)) if m else None

    def _rotate_state_dir(self):
        """Delete old state dirs, keeping the most recent ``state_keep_last_n``
        rolling entries plus any long-term anchors.

        Called from ``save_checkpoint`` AFTER the new state is written and
        fsynced, so the just-saved dir is already in ``entries`` and counts
        toward ``state_keep_last_n``.

        Two retention bands (union is preserved):
          - Rolling: most recent ``state_keep_last_n`` entries.
          - Long-term anchors: steps where step % ``state_long_term_save_every`` == 0
            AND step >= ``state_long_term_start``. Disabled when
            ``state_long_term_save_every`` is 0 (default).
        """
        if not os.path.isdir(self.state_dir):
            return
        entries = []
        for name in os.listdir(self.state_dir):
            s = self._step_from_name(name)
            full = os.path.join(self.state_dir, name)
            if s is not None and os.path.isdir(full):
                entries.append((s, full))
        entries.sort(key=lambda t: t[0])

        long_term = self.state_long_term_save_every
        long_term_start = self.state_long_term_start
        is_long_term = lambda s: (
            long_term > 0 and s % long_term == 0 and s >= long_term_start
        )

        rolling = [e for e in entries if not is_long_term(e[0])]
        cutoff = max(self.state_keep_last_n, 0)
        to_delete = rolling[:-cutoff] if cutoff > 0 else rolling
        import shutil
        for _, path in to_delete:
            logger.info("Rotating out old state dir: %s", path)
            shutil.rmtree(path, ignore_errors=True)

    def _rotate_weights_dir(self):
        """Delete old weights files, keeping the most recent
        ``weights_keep_last_n`` rolling files plus any long-term anchors
        (multiples of ``long_term_save_every``).

        Called from ``save_checkpoint`` AFTER the new weights file is written
        and fsynced, so the just-saved file is already in ``entries`` and
        counts toward ``weights_keep_last_n``.
        """
        if not os.path.isdir(self.weights_dir):
            return
        entries = []
        for name in os.listdir(self.weights_dir):
            if not name.endswith(".pt"):
                continue
            stem = name[:-3]
            s = self._step_from_name(stem)
            full = os.path.join(self.weights_dir, name)
            if s is not None and os.path.isfile(full):
                entries.append((s, full))
        entries.sort(key=lambda t: t[0])

        long_term = self.long_term_save_every
        is_long_term = lambda s: long_term > 0 and (s % long_term == 0)

        rolling = [e for e in entries if not is_long_term(e[0])]
        cutoff = max(self.weights_keep_last_n, 0)
        to_delete = rolling[:-cutoff] if cutoff > 0 else rolling
        for _, path in to_delete:
            logger.info("Rotating out old weights file: %s", path)
            try:
                os.remove(path)
            except OSError as e:
                logger.warning("Failed to remove %s: %s", path, e)

    def _drop_page_cache_for_save(self):
        """Evict Linux page cache before a checkpoint write to make headroom
        for the writeback spike.

        Long runs accumulate huge page cache (~800GB observed on the
        2TB-RAM host after a few epochs over the 75GB dataset). The ~140GB
        write spike of a state save can race the kernel's reclaim and
        trigger global host OOM that wipes the in-flight checkpoint and
        kills training (incident 2026-05-24, see FASTWAM_LTX.md §2.5).

        ``sync`` flushes dirty pages first (so nothing in-flight is lost);
        ``drop_caches=1`` then evicts only clean cache. Requires
        passwordless sudo for ``tee /proc/sys/vm/drop_caches``. Failure
        logs a warning and proceeds — save then runs as before the fix.
        """
        try:
            subprocess.run(["sync"], check=False, timeout=60)
            result = subprocess.run(
                ["sudo", "-n", "tee", "/proc/sys/vm/drop_caches"],
                input=b"1\n",
                check=False,
                capture_output=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.warning(
                    "drop_caches failed (rc=%d, stderr=%s); save proceeds "
                    "without page-cache headroom",
                    result.returncode,
                    result.stderr.decode("utf-8", errors="replace")[:200],
                )
            else:
                logger.info("drop_caches=1 issued before checkpoint save")
        except Exception as e:
            logger.warning(
                "drop_caches errored (%s); save proceeds anyway", e
            )

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        # Drop Linux page cache before writing — long runs accumulate ~800GB
        # of page cache, and the ~140GB writeback spike of a state save
        # raced kernel reclaim and triggered global host OOM (incident
        # 2026-05-24 wiped all ckpts; see FASTWAM_LTX.md §2.5). drop_caches
        # only evicts clean cache, sync flushes dirty pages first.
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self._drop_page_cache_for_save()
        self.accelerator.wait_for_everyone()

        # Save FIRST, then rotate. The previous order was rotate-then-save,
        # which lost the previous slot if save crashed mid-write — that's
        # exactly what happened at step 7500 on 2026-05-24 (step_005000 was
        # deleted before the OOM hit during step_007500 save, leaving no
        # resumable ckpt). Transient disk cost: ~2x state during the save
        # window (state_keep_last_n=2 → ~280GB), still well under disk
        # budget. See FASTWAM_LTX.md §2.5.

        # _save_weights_checkpoint now runs on all ranks (state_dict gathering
        # is a collective op under FSDP); the function itself guards disk writes
        # to main only.
        ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)

        if self._manual_optim_ckpt:
            # Manual adamw8bit + FSDP path: `accelerator.save_state` routes the
            # optimizer through `FSDP.optim_state_dict`, which crashes on the
            # block-quantized bnb 8-bit state (see __init__). Save the pieces
            # separately:
            #   * model — accelerate's own FSDP model save (DCP, the
            #     SHARDED_STATE_DICT path). This is exactly what
            #     `accelerator.save_state` does for the model and it works
            #     fine; only the optimizer half is broken. Reusing it avoids
            #     hand-rolling a state_dict over the MoT's shared-parameter
            #     structure (mot.blocks vs mot.mixtures.*._inner).
            #   * optimizer — each rank writes its own raw bnb shard via
            #     `self.optimizer.state_dict()`: rank-local, triggers no
            #     collective, so it never reaches the FSDP.optim_state_dict
            #     gather that breaks.
            #   * scheduler — rank-replicated, so only rank 0 writes it.
            from accelerate.utils.fsdp_utils import save_fsdp_model

            save_fsdp_model(
                self.accelerator.state.fsdp_plugin,
                self.accelerator,
                self.model,
                state_path,
                0,
            )
            rank = self.accelerator.process_index
            optim_path = os.path.join(state_path, f"optimizer_rank{rank}.pt")
            torch.save(self.optimizer.state_dict(), optim_path)
            if self.accelerator.is_main_process:
                torch.save(
                    self.scheduler.state_dict(),
                    os.path.join(state_path, "scheduler.pt"),
                )
                self._save_trainer_state(state_path)
            # Barrier so every rank's per-rank optimizer write has landed on
            # disk before save_checkpoint returns (rotation of this dir on the
            # next save must not race a still-writing rank).
            self.accelerator.wait_for_everyone()
        else:
            # Plain adamw / DeepSpeed / DDP path — unchanged.
            self.accelerator.save_state(output_dir=state_path)
            if self.accelerator.is_main_process:
                self._save_trainer_state(state_path)
            self.accelerator.wait_for_everyone()

        # New state is fully written across all ranks. Sync to disk before
        # rotation so the new ckpt is durable before we delete the old one
        # (os.sync flushes the whole kernel writeback queue, not just our
        # files — overkill but cheap and unambiguous).
        if self.accelerator.is_main_process:
            os.sync()
            self._rotate_state_dir()
            self._rotate_weights_dir()
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    @staticmethod
    def _is_manual_optim_ckpt_dir(state_dir: str) -> bool:
        """A manual adamw8bit+FSDP checkpoint dir is identified either by the
        `manual_optim_ckpt` marker in `trainer_state.json` or, defensively, by
        the presence of `optimizer_rank*.pt` shard files."""
        state_dir = Path(state_dir)
        state_file = state_dir / "trainer_state.json"
        if state_file.exists():
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    if bool(json.load(f).get("manual_optim_ckpt", False)):
                        return True
            except (json.JSONDecodeError, OSError):
                pass
        return any(state_dir.glob("optimizer_rank*.pt"))

    def _load_manual_training_state(self, state_dir: str):
        """Resume the manual adamw8bit+FSDP checkpoint written by
        `save_checkpoint`: accelerate FSDP sharded model state + per-rank bnb
        optimizer shards + rank-replicated scheduler. Counterpart to the manual
        save branch — it deliberately does NOT call `accelerator.load_state`
        (whose optimizer half hits the same FSDP.optim_state_dict break)."""
        state_dir = Path(state_dir)
        state_file = state_dir / "trainer_state.json"
        if not state_file.exists():
            raise FileNotFoundError(
                f"Manual-checkpoint resume requires `trainer_state.json` in {state_dir}"
            )
        with open(state_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        # Per-rank optimizer shards are only valid for the same world size.
        saved_world_size = int(payload.get("world_size", -1))
        cur_world_size = int(self.accelerator.num_processes)
        if saved_world_size != cur_world_size:
            raise RuntimeError(
                "Cannot resume manual adamw8bit+FSDP checkpoint: it was saved "
                f"with world_size={saved_world_size} but the current run has "
                f"world_size={cur_world_size}. Per-rank optimizer shards are "
                "only loadable at the identical world size."
            )

        # Model: replicate accelerate's `load_fsdp_model` SHARDED_STATE_DICT
        # path (DCP read of the `pytorch_model_fsdp_0/` dir written by
        # `save_fsdp_model`) BUT finish with a NON-strict `load_state_dict`.
        #
        # Why non-strict: a MoT block param is reachable through aliased module
        # paths (mot.blocks.<i>.<expert>_block.* — the FSDP-wrapped canonical
        # one — plus mixtures.<expert>.blocks.* and
        # mixtures.video._inner.transformer_blocks.*). Only the canonical path
        # is checkpointed (the MoT save hook drops the aliases) and only it is
        # loadable — the alias paths are FSDP placeholders that cannot receive
        # data. accelerate's `load_fsdp_model` does a *strict* load, which
        # rejects the (intentionally) missing alias keys. Non-strict tolerates
        # them; the shared nn.Module references propagate the canonical values
        # to the alias paths automatically.
        import torch.distributed.checkpoint as dist_cp
        from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        fsdp_plugin = self.accelerator.state.fsdp_plugin
        ckpt_dir = os.path.join(str(state_dir), "pytorch_model_fsdp_0")
        logger.info("Manual-checkpoint resume: restoring FSDP model state from %s", ckpt_dir)
        with FSDP.state_dict_type(
            self.model,
            fsdp_plugin.state_dict_type,
            fsdp_plugin.state_dict_config,
            fsdp_plugin.optim_state_dict_config,
        ):
            model_sd = {"model": self.model.state_dict()}
            dist_cp.load(
                state_dict=model_sd,
                storage_reader=dist_cp.FileSystemReader(ckpt_dir),
                planner=DefaultLoadPlanner(),
            )
            missing, unexpected = self.model.load_state_dict(
                model_sd["model"], strict=False
            )
        # Only the alias paths may be missing; a missing canonical block key
        # (or any unexpected key) is a real corruption.
        bad_missing = [
            k for k in missing
            if ".video_block." in k or ".action_block." in k
        ]
        if unexpected:
            raise RuntimeError(
                f"Manual-checkpoint resume: unexpected model keys "
                f"{unexpected[:6]}{'...' if len(unexpected) > 6 else ''}"
            )
        if bad_missing:
            raise RuntimeError(
                f"Manual-checkpoint resume: canonical block keys missing "
                f"{bad_missing[:6]}{'...' if len(bad_missing) > 6 else ''}"
            )

        # Optimizer: each rank loads its own bnb shard.
        rank = self.accelerator.process_index
        optim_path = state_dir / f"optimizer_rank{rank}.pt"
        if not optim_path.exists():
            raise FileNotFoundError(
                f"Manual-checkpoint resume: optimizer shard not found for rank "
                f"{rank}: {optim_path}"
            )
        self.optimizer.load_state_dict(torch.load(str(optim_path), map_location="cpu"))

        # Scheduler: rank-replicated single file.
        scheduler_path = state_dir / "scheduler.pt"
        if scheduler_path.exists():
            self.scheduler.load_state_dict(torch.load(str(scheduler_path), map_location="cpu"))
        else:
            logger.warning(
                "Manual-checkpoint resume: `scheduler.pt` missing in %s; "
                "scheduler state not restored.",
                state_dir,
            )

        # Step / epoch / batch_in_epoch — identical semantics to the
        # `accelerator.load_state` branch below.
        self.global_step = int(payload["global_step"])
        if "epoch" in payload and "batch_in_epoch" in payload:
            self.epoch = int(payload["epoch"])
            self.batch_in_epoch = int(payload["batch_in_epoch"])
            self.train_sampler.set_epoch_offset(self.epoch)
            self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
            logger.info(
                "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                self.epoch,
                self.batch_in_epoch,
                self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
            )
        else:
            self.epoch = 0
            self.batch_in_epoch = 0
            self.train_sampler.clear_resume_batch_offset()
            logger.warning(
                "Manual-checkpoint `trainer_state.json` lacks `epoch`/`batch_in_epoch`; "
                "dataloader progress resume is skipped."
            )
        self.accelerator.wait_for_everyone()
        logger.info(
            "Resumed manual adamw8bit+FSDP training state from %s at step=%d",
            state_dir,
            self.global_step,
        )

    def load_training_state(self, state_dir: str):
        # Manual adamw8bit+FSDP checkpoints are saved with a layout
        # `accelerator.load_state` cannot read (per-rank bnb optimizer shards,
        # no accelerate plan files); detect and route them to the manual loader.
        if self._is_manual_optim_ckpt_dir(state_dir):
            if not self._manual_optim_ckpt:
                raise RuntimeError(
                    f"Checkpoint dir {state_dir} is a manual adamw8bit+FSDP "
                    "checkpoint, but the current run is not configured for "
                    "adamw8bit under FSDP. Resume it with the same "
                    "optimizer_type / distributed setup."
                )
            self._load_manual_training_state(state_dir)
            return

        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def _maybe_compile_mot(self):
        """Block-level torch.compile of the MoT transformer blocks.

        Compiles each ``MoTBlock`` **in place** via ``nn.Module.compile()`` —
        NOT ``block = torch.compile(block)``. The latter replaces the block
        with a ``torch._dynamo`` ``OptimizedModule`` wrapper, which:
          * breaks ``accelerator.unwrap_model`` (``extract_model_from_parallel``
            hits ``KeyError: '_orig_mod'``),
          * stops FSDP's ``transformer_layer_cls_to_wrap=MoTBlock`` from
            matching (the block is no longer a ``MoTBlock`` instance),
          * prefixes ``state_dict`` keys with ``_orig_mod.``.
        ``nn.Module.compile()`` compiles the forward in place and leaves the
        module as a plain ``MoTBlock``, so all three problems disappear.

        Runs before ``accelerator.prepare`` so FSDP wraps the (still
        ``MoTBlock``-typed) blocks normally. No-op when there is no ``mot``.

        Dynamo/Inductor knobs (ported from LTX-2's ``compile_transformer``):
          * ``dynamic=True`` — the video token count varies per batch (frame
            count / cameras); compiling for dynamic shapes avoids a recompile
            on every new sequence length.
          * ``cache_size_limit=256`` — headroom so the handful of shape/stride
            variants the MoT block sees do not evict each other and thrash.
          * ``inline_inbuilt_nn_modules`` — fewer graph breaks at submodule
            boundaries, so Inductor can fuse across them (this is what
            collapses the ~245k tiny elementwise/reduce kernel launches the
            profiler flagged).
        """
        if not self.compile_mot:
            return
        mot = getattr(self.model, "mot", None)
        blocks = getattr(mot, "blocks", None)
        if blocks is None:
            logger.warning("compile_mot=true but model has no `mot.blocks`; skipping torch.compile.")
            return

        # Global Dynamo/Inductor config — set once, applies to every compiled
        # region. Guarded with hasattr so an older torch silently skips knobs
        # it does not have rather than crashing.
        import torch._dynamo as _dynamo
        for attr, val in (
            ("cache_size_limit", 256),
            ("inline_inbuilt_nn_modules", True),
            ("allow_unspec_int_on_nn_module", True),
        ):
            if hasattr(_dynamo.config, attr):
                setattr(_dynamo.config, attr, val)

        for blk in blocks:
            blk.compile(dynamic=True)
        logger.info(
            "torch.compile applied in-place to %d MoT blocks (dynamic=True, "
            "cache_size_limit=256).",
            len(blocks),
        )

    def _maybe_build_profiler(self):
        """Build a torch.profiler for this rank when FASTWAM_TRAINER_PROFILE=1.

        Writes chrome traces to <output_dir>/profiler/. Use this to find where
        a training step actually spends time (VAE encode vs MoT forward vs
        dataloader) before reaching for compile / kernel changes.

        Env knobs:
          FASTWAM_TRAINER_PROFILE       "1" to enable (default off).
          FASTWAM_TRAINER_PROFILE_RANKS comma-separated rank ints, or "all".
                                        Default "0" (main process only).
          FASTWAM_TRAINER_PROFILE_FREQ  steps between active traces (default 200,
                                        must exceed warmup+active = 4).

        schedule: wait → warmup(3) → active(1) → repeat. One active trace per
        FREQ steps keeps trace volume and overhead negligible.
        """
        if os.environ.get("FASTWAM_TRAINER_PROFILE", "0") != "1":
            return None

        ranks_env = os.environ.get("FASTWAM_TRAINER_PROFILE_RANKS", "0").strip().lower()
        if ranks_env == "all":
            profile_this_rank = True
        else:
            target = {int(r) for r in ranks_env.split(",") if r.strip()}
            profile_this_rank = self.accelerator.process_index in target
        if not profile_this_rank:
            return None

        freq = max(int(os.environ.get("FASTWAM_TRAINER_PROFILE_FREQ", "200")), 5)
        active, warmup = 1, 3
        wait = max(freq - active - warmup, 0)
        trace_dir = os.path.join(self.output_dir, "profiler")
        ensure_dir(trace_dir)
        logger.info(
            "torch.profiler enabled (rank %d): active trace every %d steps -> %s",
            self.accelerator.process_index,
            freq,
            trace_dir,
        )
        return torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=0),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            with_stack=False,
        )

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        profiler = self._maybe_build_profiler()
        if profiler is not None:
            profiler.start()

        while self.global_step < self.max_steps:
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                # Always invoke via __call__ so FSDP's pre-forward all-gather hook fires
                # (calling `.training_loss` directly bypasses it and leaves params sharded).
                with self.accelerator.autocast():
                    loss, loss_dict = self.model(sample)
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    # Defensive nan/inf guard: if any rank has a non-finite
                    # grad_norm, skip optimizer.step() on every rank so the
                    # master weights stay clean. We all-gather a 0/1 flag so
                    # all ranks make the same decision.
                    grad_norm_local = (
                        grad_norm.detach().float()
                        if torch.is_tensor(grad_norm)
                        else torch.tensor(float(grad_norm), device=loss.device, dtype=torch.float32)
                    ).reshape(1)
                    nonfinite_local = (~torch.isfinite(grad_norm_local)).long()
                    nonfinite_gathered = self.accelerator.gather(nonfinite_local)
                    skip_optim_step = bool(nonfinite_gathered.sum().item() > 0)
                    if skip_optim_step:
                        if self.accelerator.is_main_process:
                            logger.warning(
                                "step %d: non-finite grad_norm detected; skipping optimizer.step()",
                                self.global_step + 1,
                            )
                        self.optimizer.zero_grad(set_to_none=True)
                    else:
                        self.optimizer.step()
                        if not self.accelerator.optimizer_step_was_skipped:
                            self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    # Cross-rank loss/grad-norm reductions are only needed for
                    # logging, so pay the all-gather collectives (and the .item()
                    # device syncs they force) on log steps only — not on every
                    # optimization step. The condition is identical on all ranks
                    # (global_step is incremented in lockstep), so every rank
                    # enters the gather block together — collective-safe.
                    is_log_step = self.log_every > 0 and self.global_step % self.log_every == 0
                    if is_log_step:
                        global_loss = float(
                            self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                        )
                        global_loss_metrics = {}
                        for key, value in loss_dict.items():
                            metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                            global_loss_metrics[key] = float(
                                self.accelerator.gather(metric_tensor).mean().item()
                            )
                        grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                        global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())

                        if self.accelerator.is_main_process:
                            eta_str, steps_per_sec = self._estimate_eta()
                            description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                                self.epoch,
                                self.global_step,
                                self.max_steps,
                                global_loss,
                            )
                            if global_loss_metrics:
                                detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                                description += detail_str + " "
                            description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                                current_lr,
                                steps_per_sec,
                                steps_per_sec * self.batch_size * self.accelerator.num_processes,
                                eta_str,
                            )
                            logger.info(description)

                            wandb_payload = {
                                "train/loss": global_loss,
                                "train/grad_norm": global_grad_norm,
                                "train/lr": current_lr,
                                "performance/steps_per_sec": steps_per_sec,
                                "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            }
                            for key, value in global_loss_metrics.items():
                                wandb_payload[f"train/{key}"] = value
                            self._wandb_log(wandb_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                                self.global_step,
                                metrics["val_loss"],
                                metrics["psnr_rd"],
                                metrics["ssim_rd"],
                            )
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            logger.info(description)
                            eval_payload = {
                                "eval/val_loss": float(metrics["val_loss"]),
                                "eval/psnr_rg": float(metrics["psnr_rg"]),
                                "eval/ssim_rg": float(metrics["ssim_rg"]),
                                "eval/psnr_rd": float(metrics["psnr_rd"]),
                                "eval/ssim_rd": float(metrics["ssim_rd"]),
                                "eval/psnr_dg": float(metrics["psnr_dg"]),
                                "eval/ssim_dg": float(metrics["ssim_dg"]),
                            }
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            self._wandb_log(eval_payload)

                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    # Advance the profiler schedule once per optimization step.
                    if profiler is not None:
                        profiler.step()

                    if self.global_step >= self.max_steps:
                        if profiler is not None:
                            profiler.stop()
                        # save_every == 0 disables checkpointing entirely
                        # (smoke tests) — skip the final save too.
                        if self.save_every > 0:
                            ckpt_info = self.save_checkpoint()
                            if self.accelerator.is_main_process:
                                logger.info(
                                    "[done] max_steps reached step=%d weights=%s state=%s",
                                    self.global_step,
                                    ckpt_info["weights_path"],
                                    ckpt_info["state_path"],
                                )
                        elif self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d (save_every=0, no checkpoint)",
                                self.global_step,
                            )
                        return

        if profiler is not None:
            profiler.stop()
        if self.save_every > 0:
            ckpt_info = self.save_checkpoint()
            if self.accelerator.is_main_process:
                logger.info(
                    "[done] training finished step=%d weights=%s state=%s",
                    self.global_step,
                    ckpt_info["weights_path"],
                    ckpt_info["state_path"],
                )
        elif self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d (save_every=0, no checkpoint)",
                self.global_step,
            )
        
