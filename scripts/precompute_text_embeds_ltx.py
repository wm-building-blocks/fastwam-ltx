"""Task 13: precompute LTX text embeddings (Gemma-3-12B-it + V2 connector).

Mirrors scripts/precompute_text_embeds.py but routes prompts through
LTXTextEncoder instead of the Wan T5 encoder. Output payload is
``{"context": (T, 4096) bf16, "mask": (T,) bool}`` saved as

    ``{cache_dir}/{sha256(prompt)}.ltx23_gemma3_12b_v2connector.pt``

so the slug never collides with the Wan T5 cache slug.

Distributed: standard env-var driven torchrun (LOCAL_RANK / WORLD_SIZE /
RANK). The Gemma forward dominates wallclock; sharding prompts across
ranks gives near-linear speedup.
"""
import hashlib
import logging
import os
import uuid
import json
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.models.ltx.ltx_text_encoder import LTXTextEncoder
from fastwam.models.ltx.ltx_video_dit import load_ltx_config_from_safetensors
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging

register_default_resolvers()
logger = get_logger(__name__)

DEFAULT_CKPT_PATH = "checkpoints/Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors"
DEFAULT_GEMMA_PATH = "checkpoints/google/gemma-3-12b-it"
DEFAULT_BATCH_SIZE = 4  # Gemma-12B is big; small batch keeps memory ~ 40GB.
DEFAULT_MAX_TOKENS = 256
SLUG = "ltx23_gemma3_12b_v2connector"


def _init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return True, dist.get_rank(), dist.get_world_size(), local_rank


def _to_bool(v):
    if isinstance(v, bool): return v
    if isinstance(v, str):
        t = v.strip().lower()
        if t in {"1","true","yes","y"}: return True
        if t in {"0","false","no","n"}: return False
    raise ValueError(f"Cannot parse bool: {v}")


def _iter_dataset_nodes(node, path="data"):
    if isinstance(node, DictConfig):
        if "dataset_dirs" in node and node.get("dataset_dirs") is not None:
            yield path, node
        for k, v in node.items():
            yield from _iter_dataset_nodes(v, f"{path}.{k}")
    elif isinstance(node, ListConfig):
        for i, v in enumerate(node):
            yield from _iter_dataset_nodes(v, f"{path}[{i}]")


def _collect_dataset_settings(data_cfg):
    dataset_dirs, cache_dirs = [], []
    for p, node in _iter_dataset_nodes(data_cfg, "data"):
        raw = node.get("dataset_dirs")
        if raw is None: continue
        cdir = node.get("text_embedding_cache_dir")
        if cdir is None or not str(cdir).strip():
            raise ValueError(f"Missing text_embedding_cache_dir at {p}")
        for d in raw:
            ds = str(d)
            if ds not in dataset_dirs:
                dataset_dirs.append(ds)
        cd = Path(str(cdir)).expanduser()
        if cd not in cache_dirs:
            cache_dirs.append(cd)
    return dataset_dirs, cache_dirs


def _read_unique_prompts(dirs):
    prompts, seen = [], set()
    for d in dirs:
        p = Path(d) / "meta" / "tasks.jsonl"
        if not p.exists():
            raise FileNotFoundError(p)
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line: continue
                rec = json.loads(line)
                pr = DEFAULT_PROMPT.format(task=str(rec["task"]))
                if pr not in seen:
                    seen.add(pr); prompts.append(pr)
    return prompts


def _atomic_save(payload, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / f".{out.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, str(tmp))
    os.replace(tmp, out)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)
    is_dist, rank, world_size, local_rank = _init_distributed()
    if cfg.data is None:
        raise ValueError("`cfg.data` required.")

    dataset_dirs, cache_dirs = _collect_dataset_settings(cfg.data)
    if not cache_dirs:
        raise ValueError("No text_embedding_cache_dir under cfg.data.")

    override = cfg.get("override_instruction")
    if override:
        prompts = [DEFAULT_PROMPT.format(task=str(override))]
        logger.info("override_instruction set; encoding 1 prompt only.")
    else:
        if not dataset_dirs:
            raise ValueError("No dataset_dirs under cfg.data.")
        prompts = _read_unique_prompts(dataset_dirs)
        if rank == 0:
            logger.info(f"Found {len(prompts)} unique prompts across {len(dataset_dirs)} datasets.")
    if not prompts:
        logger.warning("Nothing to do.")
        return

    overwrite = _to_bool(cfg.get("overwrite", True))
    device = (f"cuda:{local_rank}" if is_dist else "cuda") if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    ckpt_path = str(cfg.model.get("ckpt_path", DEFAULT_CKPT_PATH))
    gemma_path = str(cfg.model.get("gemma_path", DEFAULT_GEMMA_PATH))
    max_tokens = int(cfg.get("max_tokens", DEFAULT_MAX_TOKENS))
    batch_size = int(cfg.get("batch_size", DEFAULT_BATCH_SIZE))

    if rank == 0:
        logger.info(f"ckpt={ckpt_path}  gemma={gemma_path}  device={device}  dtype={dtype}")
        logger.info(f"max_tokens={max_tokens}  batch_size={batch_size}  overwrite={overwrite}")

    config = load_ltx_config_from_safetensors(ckpt_path)
    text_encoder = LTXTextEncoder.from_config(config).to(dtype=dtype)
    miss, unexp = text_encoder.load_pretrained_safetensors(ckpt_path, strict=False)
    if miss or unexp:
        raise RuntimeError(f"LTXTextEncoder load mismatch: missing={miss[:3]}, unexpected={unexp[:3]}")
    text_encoder = text_encoder.to(device=device).eval()
    text_encoder.attach_gemma(gemma_path, torch_dtype=dtype)
    text_encoder.gemma = text_encoder.gemma.to(device)

    # Shard prompts across ranks.
    prompts_local = prompts[rank::world_size] if is_dist else prompts

    # Existence check
    if not overwrite:
        to_encode = []
        for pr in prompts_local:
            h = hashlib.sha256(pr.encode()).hexdigest()
            fn = f"{h}.{SLUG}.pt"
            if all((cd / fn).exists() for cd in cache_dirs):
                continue
            to_encode.append(pr)
        prompts_local = to_encode

    stats = {str(cd): {"new":0, "overwrite":0, "skip":0} for cd in cache_dirs}
    over_len = 0

    pbar = tqdm(
        total=len(prompts_local),
        desc=f"LTX-encode (rank {rank}/{world_size})" if is_dist else "LTX-encode",
        disable=is_dist and rank != 0,
    )
    with torch.no_grad():
        for start in range(0, len(prompts_local), batch_size):
            batch = prompts_local[start:start+batch_size]
            ctx, msk = text_encoder.encode(batch, device=torch.device(device), max_tokens=max_tokens)
            over_len += int(msk.all(dim=1).sum().item())
            for i, pr in enumerate(batch):
                h = hashlib.sha256(pr.encode()).hexdigest()
                fn = f"{h}.{SLUG}.pt"
                payload = {
                    "context": ctx[i].detach().to("cpu", torch.bfloat16).contiguous(),
                    "mask":    msk[i].detach().to("cpu", torch.bool).contiguous(),
                }
                for cd in cache_dirs:
                    out = cd / fn
                    key = str(cd)
                    if out.exists() and not overwrite:
                        stats[key]["skip"] += 1; continue
                    if out.exists():
                        stats[key]["overwrite"] += 1
                    else:
                        stats[key]["new"] += 1
                    _atomic_save(payload, out)
            pbar.update(len(batch))
    pbar.close()

    if is_dist:
        rd = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
        ov = torch.tensor([over_len], device=rd, dtype=torch.long)
        dist.all_reduce(ov, op=dist.ReduceOp.SUM)
        over_len = int(ov.item())
        cm = torch.tensor(
            [[stats[str(cd)][k] for k in ("new","overwrite","skip")] for cd in cache_dirs],
            device=rd, dtype=torch.long,
        )
        dist.all_reduce(cm, op=dist.ReduceOp.SUM)
        if rank == 0:
            for i, cd in enumerate(cache_dirs):
                k = str(cd)
                stats[k] = {"new":int(cm[i,0]),"overwrite":int(cm[i,1]),"skip":int(cm[i,2])}

    if rank == 0:
        logger.info(f"Over-length prompts (no padding @ max_tokens={max_tokens}): {over_len}")
        for cd in cache_dirs:
            k = str(cd)
            logger.info(f"Cache {k}: new={stats[k]['new']} overwrite={stats[k]['overwrite']} skip={stats[k]['skip']}")

    if is_dist:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
