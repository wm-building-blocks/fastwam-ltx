# FastWAM-LTX 在 RoboTwin 2.0 上的闭环测评运行手册

> 即使你第一次接触这套测评，跟着本文一步步做就能跑出结果。
>
> 每节都标注【在哪台机器】，命令直接复制粘贴可用。
> 涵盖：从训练机拉 ckpt → 在测评机上准备环境 + 修补丁 + 合并 ckpt + 跑测评 → 看结果。

---

## 0. 这个文档帮你完成什么

输入：训练机上的一个新 ckpt（FSDP 训练产物）。

输出：在 RoboTwin 仿真里跑 20 个 episode 的闭环测评，得到任务**成功率**（例如 `place_fan` 任务在 `demo_clean` 配置下成功了多少个）。

中间会产出：

- 每集的 rollout 视频（用于观察模型实际行为）
- 完整的 RoboTwin 日志
- 结果摘要文件 `_result_clean.txt`

整个流程在三台不同的机器之间协调；其中**所有真正的测评 GPU 工作都发生在测评机上**。

---

## 1. 三台机器是什么

| 角色 | 谁 | 这台机器上有什么 | 这台机器上要做什么 |
|---|---|---|---|
| **训练机 (171)** | `admin@208.64.254.171`（SSH 端口 22） | 训练产出的 ckpt + DCP 分片、源代码仓库 | 把 ckpt 推给测评机 |
| **测评机 (exx)** | `exx@64.62.194.199`（SSH 端口 22 或 3322 都可以） | RoboTwin 仿真资产、L40S GPU、源代码仓库 | 真正跑测评的地方 |
| **本地（你坐的电脑）** | 在本机有 `~/.ssh/` 密钥 | 两台机器的 SSH 密钥 | 协调（发命令、传文件、看结果） |

**关键点**：你**不直接**从本地跑测评。你 SSH 进测评机，在测评机上跑。

### 1.1 SSH 密钥准备（一次性）

测评机用专门的密钥（不要用通用密钥）。本地应该有 `~/.ssh/eval_exx` 文件——如果没有，跟环境管理员要。

【在本地】测试两条连接都能通：

```bash
ssh -i ~/.ssh/id_ed25519 admin@208.64.254.171 'whoami; hostname'
# 期望输出: admin / sn4622121353

ssh -i ~/.ssh/eval_exx -p 22 exx@64.62.194.199 'whoami; hostname'
# 期望输出: exx / cirrus-legwork
```

如果连不上：

- 171 用 `~/.ssh/id_ed25519`，用户名 `admin`
- 测评机用 `~/.ssh/eval_exx`，用户名 `exx`，端口 22 或 3322
- 公钥需要预先放在对应机器的 `~/.ssh/authorized_keys` 里——如果是新机器，跟运维要权限或参考"添加新机器"附录

### 1.2 让 171 能直接传文件给测评机（一次性）

合并 ckpt 时需要把 57GB DCP 从 171 直接推到测评机（绕过本地，省一半带宽）。
需要把测评机密钥放到 171 上一份：

【在本地】

```bash
scp -i ~/.ssh/id_ed25519 ~/.ssh/eval_exx admin@208.64.254.171:~/.ssh/eval_exx_key
ssh -i ~/.ssh/id_ed25519 admin@208.64.254.171 'chmod 600 ~/.ssh/eval_exx_key'
```

之后 171 上就能 `ssh -i ~/.ssh/eval_exx_key exx@64.62.194.199 ...`。

> **安全提示**：测评完成后建议把 171 上的临时密钥删掉：
> `ssh admin@208.64.254.171 'rm -f ~/.ssh/eval_exx_key'`

---

## 2. 阻塞点速查（如果你已经做过一次，下次直接看这个）

vanilla 仓库 `python experiments/robotwin/eval_robotwin_single.py ...` 直接跑会撞到的 6 个坑：

| # | 现象 | 章节 |
|---|---|---|
| 1 | `FileNotFoundError: ltx-2.3-22b-dev.safetensors` | §4 |
| 2 | `size mismatch: torch.Size([0]) ... torch.Size([4096])` | §8 |
| 3 | `Segmentation fault` in `mplib.sapien_utils.conversion` | §5.2 |
| 4 | `RuntimeError: Call attach_gemma(...) before encode()` | §6.1 |
| 5 | `CUDA OOM` 给 gemma 搬 GPU 时 | §6 |
| 6 | `ModuleNotFoundError: 'fastwam'` | §5.3 |

如果是**老手**，直接跳到 §9 看跑测评的命令。

---

# 第一部分：测评机环境（一次性搭建，所有下面的命令都在测评机上）

## 3. 进入测评机

【在本地】

```bash
ssh -i ~/.ssh/eval_exx -p 22 exx@64.62.194.199
```

进去后，你在 `exx@cirrus-legwork:~$`。

【在测评机】下面所有 §3-§8 的命令都在这里执行。

### 3.1 确认仓库位置

```bash
ls /data/aether_wam/wam/FastWAM_LTX
# 应该看到: checkpoints  configs  docs  experiments  runs  scripts  src  third_party  ...
```

如果路径不对，先找到 `FastWAM_LTX` 仓库的实际位置，下面所有 `/data/aether_wam/wam/FastWAM_LTX` 都换成实际路径。

### 3.2 确认 GPU 状态

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
```

你需要至少一张**空闲的 ≥ 40 GB 显存的卡**。下面把这张卡的编号叫做 `<GPU_ID>`，命令里 `gpu_id=2` 的 `2` 就改成它。

---

## 4. 文件系统：相对路径软链（一次性）

`eval_robotwin_single.py` 启动 RoboTwin 子进程时把 cwd 切到 `third_party/RoboTwin/`，
模型配置里的相对路径 `checkpoints/...` 从那个 cwd 解析不到仓库根的 `checkpoints/`。建一个软链解决：

```bash
cd /data/aether_wam/wam/FastWAM_LTX
ln -s "$(pwd)/checkpoints" third_party/RoboTwin/checkpoints

# 验证
ls -L third_party/RoboTwin/checkpoints/Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors
# 应该列出 46GB 那个文件，不报错
```

如果 `checkpoints/Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors` 不存在，跟环境管理员要——它是 LTX-2.3 视频基座，体积 46GB，没它跑不了。

---

## 5. conda 环境（一次性）

### 5.1 克隆现有的 `fastwam_aether` → `fastwam_ltx`

测评机已经有一个 `fastwam_aether` 环境（包齐全但 editable 链失效）。克隆它再修：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create --clone fastwam_aether --name fastwam_ltx -y
```

> 如果连 `fastwam_aether` 也没有，参考附录"从零构建环境"。

### 5.2 修关键包版本（解决两个 segfault / 不匹配）

```bash
B=~/miniconda3/envs/fastwam_ltx/bin

# numpy 2.x 会让 mplib segfault；降到 1.26.4
$B/python -m pip install --no-deps "numpy==1.26.4"

# torchaudio 默认是 2.11.0（要 CUDA 13），跟我们的 torch 2.7.1+cu128 不匹配
$B/python -m pip install --no-deps "torchaudio==2.7.1+cu128" \
    --index-url https://download.pytorch.org/whl/cu128

# ltx-core / ltx-pipelines 用 uv_build 后端，pip 装不动
$B/python -m pip install uv
```

### 5.3 修 editable 包路径

仓库以前在 `/data/aether_wam/FastWAM_LTX`，现在搬到 `/data/aether_wam/wam/FastWAM_LTX`。
旧 editable 链全部失效，重装指向新位置：

```bash
B=~/miniconda3/envs/fastwam_ltx/bin
R=/data/aether_wam/wam/FastWAM_LTX

# 主包（setuptools 后端）
$B/python -m pip install --no-deps --no-build-isolation -e $R

# ltx-2 子模块（uv_build 后端）
$B/uv pip install --python $B/python --no-deps \
    -e $R/third_party/ltx-2/packages/ltx-core \
    -e $R/third_party/ltx-2/packages/ltx-pipelines
```

### 5.4 验证

```bash
cd /data/aether_wam/wam/FastWAM_LTX
~/miniconda3/envs/fastwam_ltx/bin/python -c "
import torch, torchaudio, numpy, fastwam, ltx_core, sapien, mplib, transformers
print('torch', torch.__version__, '| numpy', numpy.__version__)
print('fastwam from', fastwam.__file__)
print('cuda ok:', torch.cuda.is_available())
"
```

期望输出（关键看 torch / numpy 版本和 fastwam 路径）：

```
torch 2.7.1+cu128 | numpy 1.26.4
fastwam from /data/aether_wam/wam/FastWAM_LTX/src/fastwam/__init__.py
cuda ok: True
```

> **不要试图 `import ltx_pipelines`**——`multigpu/` 子目录在当前 checkout 里缺失，但仓库代码不需要它。

---

## 6. 源码补丁（4 处，一次性）

vanilla 仓库需要 4 处修改才能完成闭环测评。每改一个文件前先备份：

```bash
cd /data/aether_wam/wam/FastWAM_LTX
cp third_party/RoboTwin/policy/fastwam_policy/deploy_policy.py{,.bak}
cp src/fastwam/models/ltx/helpers/loader.py{,.bak}
cp src/fastwam/models/ltx/fastwam.py{,.bak}
```

### 6.1 强制 attach gemma

文件：`third_party/RoboTwin/policy/fastwam_policy/deploy_policy.py`

在 `WorldActionRobotWinPolicy.__init__` 中（搜索 `model_cfg_copy.load_text_encoder = True`），在它**正下面**加一行：

```python
model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
model_cfg_copy.load_text_encoder = True
model_cfg_copy.attach_gemma_to_text_encoder = True   # ← 新增这一行

self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
```

不加：测评推理时报 `RuntimeError: Call attach_gemma(gemma_root) before encode()`。

### 6.2 让 text_encoder 永远在 CPU

文件：`src/fastwam/models/ltx/helpers/loader.py`

L40S 46GB 装不下 mot(~28GB) + gemma(24GB) 同卡。让整个 text_encoder（含 gemma）留 CPU：

把下面这一行**注释掉**：

```python
text_encoder = text_encoder.to(device=device)
```

再把这个 if 整块**注释掉**：

```python
if device != "cpu":
    text_encoder.gemma = text_encoder.gemma.to(device)
```

不改：`CUDA out of memory` 加载 gemma 时。

### 6.3 FastWAM.to() 不能递归 text_encoder

文件：`src/fastwam/models/ltx/fastwam.py`

`FastWAM.to(device)` 默认通过 `super().to()` 把所有子模块（包括 text_encoder）拖到 GPU。
把它替换成：

```python
def to(self, *args, **kwargs):
    # Keep text_encoder (incl. Gemma 12B) on CPU regardless of target device.
    saved_te = self._modules.pop("text_encoder", None)
    super().to(*args, **kwargs)
    self.mot.to(*args, **kwargs)
    if saved_te is not None:
        self._modules["text_encoder"] = saved_te
    self.vae.to(*args, **kwargs)
    return self
```

### 6.4 `encode_prompt` 加缓存 + 用 te_device

同一文件 `src/fastwam/models/ltx/fastwam.py`，替换 `encode_prompt`：

```python
@torch.no_grad()
def encode_prompt(self, prompt):
    if self.text_encoder is None:
        raise ValueError(
            "Prompt encoding requires loaded LTXTextEncoder (with Gemma attached). "
            "Either set `load_text_encoder=true` and `attach_gemma_to_text_encoder=true`, "
            "or provide precomputed `context/context_mask` from the cache."
        )
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    cache_key = tuple(prompts)
    if not hasattr(self, "_encode_prompt_cache"):
        self._encode_prompt_cache = {}
    if cache_key in self._encode_prompt_cache:
        emb, msk = self._encode_prompt_cache[cache_key]
        return (emb.to(device=self.device, dtype=self.torch_dtype),
                msk.to(device=self.device, dtype=torch.bool))
    # 在 text_encoder 实际所在的设备上做编码（CPU）
    te_device = next(self.text_encoder.parameters()).device
    prompt_emb, binary_mask = self.text_encoder.encode(prompts, device=te_device)
    self._encode_prompt_cache[cache_key] = (prompt_emb.detach().cpu(),
                                             binary_mask.detach().cpu())
    return (prompt_emb.to(device=self.device, dtype=self.torch_dtype),
            binary_mask.to(device=self.device, dtype=torch.bool))
```

> 每集 paraphrase 不同，但同集内多次 replan 会命中缓存。
> 单次 cache miss 的 CPU gemma 前向 ~30-60 s（256 tokens, 12B params, bf16）。

---

## 7. Gemma 模型放置（一次性）

如果 `checkpoints/google/gemma-3-12b-it-qat-q4_0-unquantized/` 不存在，从 171 拉过来（23GB）。

**先**完成 §1.2（让 171 能 ssh 到测评机）。然后【在 171】：

```bash
ssh -i ~/.ssh/id_ed25519 admin@208.64.254.171 'bash -s' <<'REMOTE'
rsync -a --info=progress2 \
  -e "ssh -p 22 -i ~/.ssh/eval_exx_key -o StrictHostKeyChecking=accept-new" \
  /home/admin/fang/FastWAM_LTX/checkpoints/google/gemma-3-12b-it-qat-q4_0-unquantized \
  exx@64.62.194.199:/data/aether_wam/wam/FastWAM_LTX/checkpoints/google/
REMOTE
```

23 GB，内网 5-10 分钟。完成后回测评机验证：

```bash
ssh -i ~/.ssh/eval_exx -p 22 exx@64.62.194.199 'du -sh /data/aether_wam/wam/FastWAM_LTX/checkpoints/google/gemma-3-12b-it-qat-q4_0-unquantized'
# 期望: 23G
```

---

至此**一次性环境搭建完成**。每次有新 ckpt 时只需做 §8-§10。

---

# 第二部分：每个新 ckpt 的完整工作流

每次训练侧产出新 ckpt（例如 `step_5000.pt`），按下面三步走。

## 8. 从训练机同步必要文件

【在本地】先决定要拉哪一个 ckpt：

```bash
# 这里改成你的实际值
TASK=place_fan_overfit       # 训练任务名（runs/ 下的目录名）
DATE=2026-05-22_10-54-52     # 训练 run 的日期目录
STEP=5000                    # 想测评哪一步
```

【在本地】触发 171 → 测评机的同步：

```bash
ssh -i ~/.ssh/id_ed25519 admin@208.64.254.171 "bash -s" <<REMOTE
TASK='$TASK'
DATE='$DATE'
STEP=$STEP
STEPDIR=step_\$(printf '%06d' \$STEP)
RUN=/home/admin/fang/FastWAM_LTX/runs/\$TASK/\$DATE
REMOTE_RUN=/data/aether_wam/wam/FastWAM_LTX/runs/\$TASK/\$DATE

# 检查源端文件确实存在
ls \$RUN/dataset_stats.json
ls \$RUN/checkpoints/state/\$STEPDIR/pytorch_model_fsdp_0/.metadata

# 在测评机端预建目录
ssh -p 22 -i ~/.ssh/eval_exx_key -o StrictHostKeyChecking=accept-new exx@64.62.194.199 \\
    "mkdir -p \$REMOTE_RUN/checkpoints/state/\$STEPDIR \$REMOTE_RUN/checkpoints/weights"

# 1) dataset_stats + config (~100KB)
rsync -a --info=progress2 -e "ssh -p 22 -i ~/.ssh/eval_exx_key" \\
    \$RUN/dataset_stats.json \$RUN/config.yaml \\
    exx@64.62.194.199:\$REMOTE_RUN/

# 2) DCP 分片 (~57GB，最大的部分)
rsync -a --info=progress2 -e "ssh -p 22 -i ~/.ssh/eval_exx_key" \\
    \$RUN/checkpoints/state/\$STEPDIR/pytorch_model_fsdp_0/ \\
    exx@64.62.194.199:\$REMOTE_RUN/checkpoints/state/\$STEPDIR/pytorch_model_fsdp_0/
REMOTE
```

**只传这些**，**不传**以下文件（省一半带宽，~60 GB）：

- `weights/step_NNNN.pt`（~55 GB）：FSDP 保存的 ckpt 是坏的（参数为空或扁平），§9 会从 DCP 重建一份正确的
- `state/.../optimizer_rank*.pt`（每个 ~4 GB × 7 = ~28 GB）：只有 resume 训练才用
- `eval/*.mp4`：训练采样视频，跟闭环测评无关

内网 rsync 57GB 通常 5-15 分钟。断网了重跑即可，rsync 会跳过已传输的文件。

---

## 9. 在测评机上合并 ckpt（从 DCP 重建）

【在本地】SSH 进测评机并起 tmux（很关键，不开 tmux 会被 SSH 断开杀进程）：

```bash
ssh -t -i ~/.ssh/eval_exx -p 22 exx@64.62.194.199 'tmux new -s consolidate'
```

【在测评机的 tmux 会话内】写合并脚本并跑：

```bash
TASK=place_fan_overfit       # 同 §8
DATE=2026-05-22_10-54-52
STEP=5000

cat > /tmp/consolidate_ckpt.py <<'PY'
"""Consolidate FastWAM-LTX FSDP DCP -> plain .pt payload.

Reads <RUN>/checkpoints/state/<STEPDIR>/pytorch_model_fsdp_0/ (7 .distcp shards)
and writes <RUN>/checkpoints/weights/<STEPDIR>_consolidated.pt.

Handles two DCP layouts seen in this repo:
  (a) flat top-level keys: model.mot.* / model.proprio_encoder.* (e.g. step_5000)
  (b) nested under a single "model" entry: {"model": {"mot.*": ..., ...}} (e.g. step_8000)
"""
import os
import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

RUN  = os.environ["RUN"]
STEP = int(os.environ["STEP"])
STEPDIR = f"step_{STEP:06d}"
DCP = f"{RUN}/checkpoints/state/{STEPDIR}/pytorch_model_fsdp_0"
OUT = f"{RUN}/checkpoints/weights/{STEPDIR}_consolidated.pt"
TMP = "/home/exx/_dcp_flat.pt"

assert os.path.isdir(DCP), f"DCP dir missing: {DCP}"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

if not os.path.exists(TMP):
    print(f"[1/3] dcp_to_torch_save ({DCP}) ...", flush=True)
    dcp_to_torch_save(DCP, TMP)
else:
    print(f"[1/3] reuse existing scratch: {TMP}", flush=True)

print(f"[2/3] loading + splitting ...", flush=True)
flat = torch.load(TMP, map_location="cpu", weights_only=False, mmap=True)

# Accept both layouts: flat (model.mot.*) and nested ({"model": {"mot.*": ...}}).
if "model" in flat and isinstance(flat["model"], dict):
    items = flat["model"].items()
else:
    items = flat.items()

mot, proprio = {}, {}
for k, v in items:
    if k.startswith("mot."):
        mot[k[len("mot."):]] = v
    elif k.startswith("proprio_encoder."):
        proprio[k[len("proprio_encoder."):]] = v
    elif k.startswith("model.mot."):
        mot[k[len("model.mot."):]] = v
    elif k.startswith("model.proprio_encoder."):
        proprio[k[len("model.proprio_encoder."):]] = v

ae = mot.get("mixtures.action.action_encoder.weight")
pp = mot.get("mixtures.video._inner.patchify_proj.weight")
empty = [k for k, v in mot.items() if hasattr(v, "numel") and v.numel() == 0]
print(f"  mot={len(mot)} proprio={len(proprio)} empty={len(empty)}")
print(f"  action_encoder.weight: {tuple(ae.shape) if ae is not None else None}")
print(f"  patchify_proj.weight: {tuple(pp.shape) if pp is not None else None}")
assert len(empty) == 0, f"consolidation still has {len(empty)} empty params"
assert ae is not None and ae.dim() == 2, "action_encoder missing/malformed"
assert pp is not None and pp.dim() == 2, "patchify_proj still flattened"

payload = {
    "mot": mot,
    "step": STEP,
    "torch_dtype": os.environ.get("CKPT_TORCH_DTYPE", "torch.bfloat16"),
}
if proprio:
    payload["proprio_encoder"] = proprio

print(f"[3/3] saving -> {OUT}", flush=True)
torch.save(payload, OUT)
print(f"DONE  ({os.path.getsize(OUT)/1e9:.1f} GB)")
PY

export RUN=/data/aether_wam/wam/FastWAM_LTX/runs/$TASK/$DATE
export STEP=$STEP
~/miniconda3/envs/fastwam_ltx/bin/python -u /tmp/consolidate_ckpt.py 2>&1 | tee /tmp/consolidate.log
```

需要 ≥ 60 GB 内存 + ~120 GB 临时磁盘空间。测评机 503 GB RAM + 896 GB `/home` 足够。
跑 3-5 分钟，期望输出末尾：

```
[3/3] saving -> /data/.../checkpoints/weights/step_005000_consolidated.pt
DONE  (55.8 GB)
```

跑完后 tmux 退出（`Ctrl+B` 然后按 `d`），或直接关掉。

**善后**：删 57 GB 暂存文件：

```bash
rm /home/exx/_dcp_flat.pt
```

> **不要**用 `nohup python ... &`，SSH 断开会杀进程导致 log 为空、产物缺失。必须用 tmux 或保活 ssh。

---

## 10. 跑测评

【在本地】重新 SSH 进测评机起一个新 tmux（脚本要跑 30-60 分钟）：

```bash
ssh -t -i ~/.ssh/eval_exx -p 22 exx@64.62.194.199 'tmux new -s eval'
```

【在测评机 tmux 内】：

```bash
cd /data/aether_wam/wam/FastWAM_LTX
source ~/miniconda3/etc/profile.d/conda.sh && conda activate fastwam_ltx

TASK=place_fan_overfit              # 同 §8
DATE=2026-05-22_10-54-52
STEP=5000
STEPDIR=step_$(printf '%06d' $STEP)
GPU_ID=2                            # 改成你 §3.2 选的那张卡

PYTHONFAULTHANDLER=1 python -u experiments/robotwin/eval_robotwin_single.py \
    ckpt=runs/$TASK/$DATE/checkpoints/weights/${STEPDIR}_consolidated.pt \
    EVALUATION.task_name=place_fan \
    EVALUATION.task_config=demo_clean \
    EVALUATION.eval_num_episodes=20 \
    EVALUATION.replan_steps=24 \
    EVALUATION.num_inference_steps=10 \
    gpu_id=$GPU_ID \
    2>&1 | tee /tmp/eval_${TASK}_${STEP}.log
```

`Ctrl+B` 然后按 `d` 分离 tmux；ssh 断开不影响。

### 10.1 关键参数怎么选

| 参数 | 含义 | 怎么选 |
|---|---|---|
| `EVALUATION.task_name` | RoboTwin 任务名 | 训练用哪个任务就填哪个 |
| `EVALUATION.task_config` | 仿真场景配置 | overfit sanity 用 `demo_clean`（同训练分布）；泛化测试用 `demo_randomized` |
| `EVALUATION.eval_num_episodes` | 跑几集 | sanity 用 20，正式用 50 / 100 |
| `EVALUATION.replan_steps` | 每隔多少仿真步 replan | 24（默认） |
| `EVALUATION.num_inference_steps` | flow-matching 采样步数 | 10（默认） |
| `gpu_id` | 用哪张 GPU | §3.2 看哪张 ≥ 40 GB 空闲 |

### 10.2 第一次跑会慢的原因

- **curobo CUDA kernel 首次 JIT 编译**：~10-20 分钟。编完缓存到 `~/.cache/torch_extensions/`，下次直接跳过。
- **LTX-2.3 基座加载**：~2 分钟（46 GB safetensors）。
- **首次 Gemma CPU 前向**：~30-60 s（256 tokens, 12B params, bf16）。

后续重跑（同 env、同 ckpt）：模型加载 2 分钟 + 每集 ~30 s = 20 集约 10-15 分钟。

### 10.3 监控进度

【在本地】另开一个终端：

```bash
ssh -i ~/.ssh/eval_exx -p 22 exx@64.62.194.199 'grep "Success rate" /tmp/eval_place_fan_overfit_5000.log | tail -5'
ssh -i ~/.ssh/eval_exx -p 22 exx@64.62.194.199 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader'
```

或【在本地】attach 回 tmux 看实时输出：

```bash
ssh -t -i ~/.ssh/eval_exx -p 22 exx@64.62.194.199 'tmux attach -t eval'
```

---

## 11. 结果文件在哪、怎么看

测评跑完后，结果在测评机的：

```
/data/aether_wam/wam/FastWAM_LTX/evaluate_results/robotwin/<task>_<date>/<run_ts>/
├── eval_<task_name>_<datetime>.log         # 完整测评日志
└── <task_name>/
    ├── _result_clean.txt                    # 关键：成功率文件
    └── episode<N>_randomized-<bool>_success-<bool>.mp4   # 20 集 rollout 视频
```

【在本地】查成功率：

```bash
ssh -i ~/.ssh/eval_exx -p 22 exx@64.62.194.199 'cat /data/aether_wam/wam/FastWAM_LTX/evaluate_results/robotwin/<task>_<date>/<run_ts>/<task_name>/_result_clean.txt'
```

格式：

```
Timestamp: 20260522_225308
Instruction Type: unseen
0.05      ← 成功率（这里 5%）
```

### 11.1 把几集视频拉本地看

【在本地】（无需本地装 ffmpeg；用测评机的 python 抽帧）：

```bash
ssh -i ~/.ssh/eval_exx -p 22 exx@64.62.194.199 'bash -s' <<'REMOTE'
PYENV=~/miniconda3/envs/fastwam_ltx/bin/python
D=/data/aether_wam/wam/FastWAM_LTX/evaluate_results/robotwin/<task>_<date>/<run_ts>/<task_name>
mkdir -p /tmp/eval_frames
$PYENV - <<EOF
import imageio.v3 as iio, numpy as np, os, glob
from PIL import Image
mp4s = sorted(glob.glob("$D/episode*.mp4"))[:3]
for f in mp4s:
    frames = iio.imread(f)
    name = os.path.basename(f).split("_")[0]   # episodeN
    for k, i in enumerate(np.linspace(0, frames.shape[0]-1, 6).astype(int)):
        Image.fromarray(frames[i]).save(f"/tmp/eval_frames/{name}_f{k}.jpg", quality=88)
print("done")
EOF
REMOTE
scp -i ~/.ssh/eval_exx -P 22 'exx@64.62.194.199:/tmp/eval_frames/*.jpg' .
```

每集得到 6 张关键帧（开始、中段、结束）。

视频布局（480×640）：

- 左上：第三人称俯视
- 右上：通常未启用（黑屏）
- 左下：另一视角
- 右下：腕部相机

---

## 12. 怎么解读结果

简版决策树：

- **demo_clean 成功率 > 30%** → overfit 成立、闭环正确，可以继续做泛化测试 `demo_randomized`。
- **demo_clean 成功率 ≈ 0** → 按下面顺序排查：

  1. **ckpt 合并是否正确**（§9 sanity check 应该已经过；如果 assert 失败说明 DCP 有问题，找训练侧）。
  2. **`dataset_stats.json` 训练/测评一致**：
     ```bash
     ssh admin@171 'md5sum /home/admin/fang/FastWAM_LTX/runs/.../dataset_stats.json'
     ssh exx@64.62.194.199 'md5sum /data/aether_wam/wam/FastWAM_LTX/runs/.../dataset_stats.json'
     ```
     md5 应该一致。
  3. **训练侧 eval mp4**（`runs/.../eval/step_*.mp4`）：模型的开环采样是否能复现行为。
     如果开环采样里 pick-and-place 也没出来 → 是训练问题（loss 没真到地板），不是测评管线问题。
  4. **`train.log` 里 `loss_action`** 最后是不是 ~0.02（很可能是个假平台）。真正 overfit 收敛应到 < ~0.005。

> 经验：MSE 0.02 ≈ 归一化空间的预测误差 ~0.14。夹爪通道 std~0.46、范围 [0,1]，
> 0.14 的误差足够让夹爪输出"半闭"，表现为"靠近物体但抓不起来"。

---

## 13. 故障速查

| 报错 | 处理 |
|---|---|
| `FileNotFoundError: ltx-2.3-22b-dev.safetensors` | 软链没建 → §4 |
| `size mismatch: torch.Size([0])` | ckpt 没合并 → §9 |
| `RuntimeError: Call attach_gemma(...) before encode()` | deploy_policy 没打补丁 → §6.1 |
| `CUDA OOM` 加载 gemma | loader.py 没改成 CPU → §6.2 |
| Segfault in `mplib.sapien_utils.conversion` | numpy 是 2.x → §5.2 |
| `libcudart.so.13: cannot open` | torchaudio 版本不匹配 → §5.2 |
| `Cannot import 'uv_build'` | 缺 uv → `pip install uv` |
| `ModuleNotFoundError: 'fastwam'` | editable 链失效 → §5.3 |
| `dataset_stats.json not found` | §8 没同步过来；或显式传 `EVALUATION.dataset_stats_path=...` |
| ssh 断开后进程死掉、log 空 | 用 tmux（§9, §10），**不要** `nohup ... &` 后立刻退出 |
| consolidate 脚本立即退出、log 空 | 远程脚本上传可能截断；`wc -l /tmp/consolidate_ckpt.py` 验证非零 |
| 测评卡在 "kinematics_fused_cu compiling" 很久 | 正常，curobo 首次 JIT 编译要 10-20 分钟；编完会缓存 |

---

## 14. 测评机改动清单（vs vanilla 仓库）

| 文件 / 路径 | 操作 | 章节 |
|---|---|---|
| `third_party/RoboTwin/checkpoints` | 新建软链 → `../../checkpoints` | §4 |
| `third_party/RoboTwin/policy/fastwam_policy/deploy_policy.py` | 加一行 `attach_gemma_to_text_encoder = True` | §6.1 |
| `src/fastwam/models/ltx/helpers/loader.py` | 跳过 text_encoder / gemma `.to(device)` | §6.2 |
| `src/fastwam/models/ltx/fastwam.py` | `.to()` 摘 text_encoder；`encode_prompt` 加缓存 | §6.3-§6.4 |
| `checkpoints/google/gemma-3-12b-it-qat-q4_0-unquantized/` | 从 171 rsync | §7 |
| `runs/<task>/<date>/...` | 每次新 ckpt 从 171 同步（最小集） | §8 |
| `runs/.../weights/step_NNNN_consolidated.pt` | 从 DCP 合并产物 | §9 |
| conda env `fastwam_ltx` | 克隆自 `fastwam_aether` + 修包版本 + 重装 editable | §5 |

源码改动都在原文件旁保留了 `.bak` 备份。

---

## 15. 附录：训练侧需要修的 bug（参考，不在本手册范围）

`trainer.py` 的 `_save_weights_checkpoint` 用 `FSDP.state_dict_type(self.model, FULL_STATE_DICT, ...)`
上下文时，对嵌套在 `mot` 下的 MoT FSDP 单元的 gather 没真正生效，rank-0 落盘的
是扁平 `FlatParameter` + 部分空张量。

修好后 `weights/step_NNNN.pt` 直接可用，§9 整节可以跳过。在那之前，每次都从 DCP 合并。
