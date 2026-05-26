# FastWAM-LTX 过拟合实验：place_fan (demo_clean)

目的：在极小数据集上验证 FastWAM-LTX 的模型 + 训练管线正确性 —— loss 能否下降、
模型能否记住训练片段。这是 correctness check，不是正式训练。

## 1. 数据

任务 `place_fan`：单臂把扇子放到垫子上 —— RoboTwin 2.0 中**最简单**的任务
(episode 帧长最短,平均 ~146 帧;单臂、单物体、pick-and-place)。

数据来自 `data/robotwin2.0/robotwin2.0`(预处理后的 LeRobot v2.1 扁平数据集,
27500 episodes,不保留 RoboTwin 原始的 `<task>/demo_clean` 目录结构)。

### clean / randomized 边界

place_fan 的 episode 区间是 **16500-17049**,按 RoboTwin 2.0 标准切法
**50 条 demo_clean + 500 条 demo_randomized**:

| episode 区间 | 模式 | 特征 |
|---|---|---|
| **16500 – 16549** | demo_clean | 纯白干净台面,无杂物/无随机背景/无随机光照 |
| 16550 – 17049 | demo_randomized | cluttered table + 纹理桌面 + 随机背景 |

边界经逐帧抽样确认,精确落在 16550(16549 仍 clean,16550 起 randomized)。
本实验**只用 demo_clean**,所以可用 episode 上限是 16500-16549 这 50 条。

> 来源参考:FastWAM 仓库 `data/robotwin2.0/keep_manifest/per_task/` 过滤出 10 个
> 任务,每个 ~550 条(50 clean + ~480 randomized + 少量缺失)。

## 2. episode 白名单机制(本实验新增)

`BaseLerobotDataset` / `RobotVideoDataset` 新增两个参数
(`src/fastwam/datasets/lerobot/`,从 FastWAM 仓库移植):

- `keep_episodes_path` —— 白名单文件,一行一个 `episode_index`;在 train/val
  split **之前**过滤,所以 `val_set_proportion` 是从过滤后的集合里取的。
- `keep_episodes_limit` —— 只取白名单文件**前 N 行**(manifest 顺序)。让一份
  50 行的 manifest 既能驱动 50 集训练,也能 `limit=20` 驱动 20 集快速实验。

manifest 文件:`data/robotwin2.0/place_fan_clean_50ep.txt` —— 50 行,16500-16549。

## 3. 配置

task config:`configs/task/place_fan_overfit.yaml`。

| 项 | 值 | 理由 |
|---|---|---|
| `keep_episodes_path` | place_fan_clean_50ep.txt | 50 条 clean manifest |
| `keep_episodes_limit` | 20 | 只用前 20 条(16500-16519),反馈更快 |
| `val_set_proportion` | 0 | 不留 val;eval 直接在这 20 条上跑 = 记忆度检验 |
| `gradient_accumulation_steps` | 1 | 有效 batch 2×7=14;20 条数据喂不满 batch 112 |
| `max_steps` | 2000(config) / 1500(首跑 CLI override) | 见 §4 |
| `learning_rate` | 1e-4,`lr_scheduler_type: constant` | 纯 overfit 不退火,loss 曲线好读 |
| `save_every` / `eval_every` | 250 / 500 | 中途出 checkpoint + 采样确认复现 |
| `compile_mot` | false | FSDP 下首步编译 ~20min 且步时收益≈0(见 FASTWAM_LTX.md §2.3) |
| `optimizer_type` | adamw8bit | |

## 4. 步数建议

flow-matching/扩散训练每步只采一个随机 timestep,单个窗口要在不同噪声水平下被
反复看很多次才能记住去噪,所以即使数据极小也需要几千步。overfit 步数对数据量是
**次线性**关系:

| 数据量 | 建议步数 |
|---|---|
| 50 clean episodes | ~3000 步起步,5000 步更稳 |
| 20 clean episodes | ~2000 步(首跑先 1500 看趋势) |

按 ~7s/step:1500 步 ≈ 3 小时,5000 步 ≈ 10 小时。

实际采用:50 条全量 clean,`max_steps=5000`(CLI override `keep_episodes_limit=50`)。

## 5. 成功判据

扩散 loss 有噪声、不会干净归零,看趋势:

- training `loss` 从初值明显下行并在低位 plateau;
- `loss_action` 和 `loss_video` **两个分支都要降**(world-action 模型);
- eval 采样出的视频能复现 place_fan 的动作 / 物体;
- 20 条数据下,预计 step 500-1000 之间出现明显塌陷信号。

## 6. 启动命令

```bash
cd /home/admin/fang/FastWAM_LTX
source ~/miniconda3/etc/profile.d/conda.sh && conda activate fastwam_ltx
export WANDB_API_KEY=<key>          # 须在 shell 里 export,不写进仓库

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 nohup bash scripts/train_fsdp.sh 7 \
    task=place_fan_overfit max_steps=1500 \
    wandb.enabled=true wandb.name=place_fan_overfit \
    > /tmp/overfit_run.log 2>&1 &
```

- 日志:`/tmp/overfit_run.log`;wandb:project `fastwam_ltx`,run `place_fan_overfit`。
- 想用全部 50 条 clean:去掉 `keep_episodes_limit`(或设 50),`max_steps` 提到 3000。

## 7. 运行记录

| 日期 | 数据 | 步数 | 结果 |
|---|---|---|---|
| 2026-05-22 | 20 clean ep (16500-16519) | 1500 | 中断 —— 头 20 步 loss 3.55→2.68 趋势健康,随后 171 失联,run 状态未确认 |
| 2026-05-22 | 50 clean ep (16500-16549) | 5000 | 收敛。loss_action 3250步起到地板(~0.012-0.025),loss_video 3250步起在 0.20-0.28 噪声振荡不再下降 —— ~3000 步即 plateau,后 2000 步白磨。ckpt:`runs/place_fan_overfit/2026-05-22_10-54-52/checkpoints/weights/step_005000.pt`。最终判据待 eval 采样 / RoboTwin 测评确认 |
