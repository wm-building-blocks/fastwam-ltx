import hashlib
import os
from collections import OrderedDict
from typing import Optional
import time
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F
from contextlib import contextmanager

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from fastwam.utils.logging_config import get_logger
from fastwam.utils import misc, pytorch_utils
from accelerate import PartialState
logger = get_logger(__name__)

_DEFAULT_TEXT_ENCODER_MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B"
_DEFAULT_TOKENIZER_MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B"


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,  # deprecated; on-the-fly T5 is used now
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        text_encoder_model_id: str = _DEFAULT_TEXT_ENCODER_MODEL_ID,
        tokenizer_model_id: str = _DEFAULT_TOKENIZER_MODEL_ID,
        text_encoder_cache_size: int = 8192,
        text_encoder_redirect_common_files: bool = True,
    ):
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
        )
    
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        
        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction

        # On-the-fly T5 text encoder lives on CPU. Initialized lazily inside
        # each dataloader worker process so we don't pay the load cost in the
        # parent process and so each worker has its own LRU.
        self.text_encoder_model_id = text_encoder_model_id
        self.tokenizer_model_id = tokenizer_model_id
        self.text_encoder_cache_size = int(text_encoder_cache_size)
        self.text_encoder_redirect_common_files = bool(text_encoder_redirect_common_files)
        self._text_encoder = None
        self._tokenizer = None
        self._text_cache: "OrderedDict[str, tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)
        
    def __len__(self):
        return len(self.lerobot_dataset)

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))
        
        image_is_pad = sample["image_is_pad"]

        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        num_cameras = 1
        if video.ndim == 5:
            video = video[:, self.video_sample_indices, :, :, :] # [num_cameras, T_video, C, H, W]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            video = video[self.video_sample_indices, :, :, :] # [T_video, C, H, W]
            T_video, C, H, W = video.shape
        image_is_pad = image_is_pad[self.video_sample_indices]

        video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 256, 320]
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T_video, C, 128, 320]
            video = torch.cat([cam_top, bottom], dim=-2)  # [T_video, C, 384, 320]
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)  # [T_video, C, H, num_cameras*W]
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)  # [T_video, C, num_cameras*H, W]
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)  # [T_video, C, H, W]

        # final resize and normalization
        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)  # [T_video, C, H, W]

        video = video.permute(1, 0, 2, 3) # [C, T_video, H, W], range [-1, 1]

        # Proxy (from lerobot): 
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :] # [T-1, state_dim]， to align with action
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}"
            )

        task = sample["instruction"]
        
        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        context, context_mask = self._get_cached_text_context(instruction)
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        
        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        return data

    def _ensure_text_encoder(self):
        if self._text_encoder is not None:
            return
        # Local imports keep the parent process free of the T5 weights and
        # also avoid an import cycle at module load time.
        from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
        from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer

        _, text_config, _, tokenizer_config = _resolve_configs(
            model_id=self.text_encoder_model_id,
            tokenizer_model_id=self.tokenizer_model_id,
            redirect_common_files=self.text_encoder_redirect_common_files,
        )
        text_config.download_if_necessary()
        tokenizer_config.download_if_necessary()

        text_encoder = _load_registered_model(
            text_config.path,
            "wan_video_text_encoder",
            torch_dtype=torch.bfloat16,
            device="cpu",
        ).eval()
        for p in text_encoder.parameters():
            p.requires_grad = False
        # Hard assert: T5 must stay on CPU so it never touches the training
        # GPU memory budget validated in Step B (~79GB peak).
        for p in text_encoder.parameters():
            assert p.device.type == "cpu", (
                f"Text encoder unexpectedly on {p.device}; must remain on CPU."
            )

        self._text_encoder = text_encoder
        self._tokenizer = HuggingfaceTokenizer(
            name=tokenizer_config.path,
            seq_len=self.context_len,
            clean="whitespace",
        )
        logger.info(
            "Loaded on-the-fly T5 text encoder on CPU (worker pid=%d, lru=%d).",
            os.getpid(),
            self.text_encoder_cache_size,
        )

    def _encode_prompt_cpu(self, prompt: str):
        self._ensure_text_encoder()
        ids, mask = self._tokenizer([prompt], return_mask=True, add_special_tokens=True)
        mask = mask.to(dtype=torch.bool)
        with torch.no_grad():
            context = self._text_encoder(ids, mask)
        context = context[0].detach().to(dtype=torch.bfloat16).contiguous()
        return context, mask[0].clone()

    def _get_cached_text_context(self, prompt: str):
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cached = self._text_cache.get(hashed)
        if cached is not None:
            self._text_cache.move_to_end(hashed)
            context, context_mask = cached
            return context, context_mask

        # Disk-cache fast path (populated by scripts/precompute_text_embeds.py).
        # When the .pt exists we get ~5x throughput vs CPU on-the-fly encode.
        if self.text_embedding_cache_dir is not None:
            cache_path = os.path.join(
                self.text_embedding_cache_dir,
                f"{hashed}.t5_len{self.context_len}.wan22t2va14b.pt",
            )
            if os.path.exists(cache_path):
                payload = torch.load(cache_path, map_location="cpu")
                context = payload["context"]
                context_mask = payload["mask"].bool()
                self._text_cache[hashed] = (context, context_mask)
                if len(self._text_cache) > self.text_encoder_cache_size:
                    self._text_cache.popitem(last=False)
                return context, context_mask

        context, context_mask = self._encode_prompt_cpu(prompt)
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Encoded context_len mismatch: expected {self.context_len}, got {context.shape[0]}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Encoded mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]}"
            )

        self._text_cache[hashed] = (context, context_mask)
        if len(self._text_cache) > self.text_encoder_cache_size:
            self._text_cache.popitem(last=False)
        return context, context_mask

    def __getstate__(self):
        # Avoid pickling the T5 weights / LRU when DataLoader workers fork-
        # spawn this dataset. Each worker re-loads T5 lazily in its own
        # process, so its memory stays process-local.
        state = self.__dict__.copy()
        state["_text_encoder"] = None
        state["_tokenizer"] = None
        state["_text_cache"] = OrderedDict()
        return state

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
