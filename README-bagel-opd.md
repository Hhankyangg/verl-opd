# BAGEL OPD 快速部署说明

这个仓库里已经迁移了两条 BAGEL OPD 实验链路：

- `BAGEL teacher -> Qwen2.5-VL student`：`zoo_scripts/zoo_run_opd_bagel2qwen_mintcot.sh`
- `Qwen2.5-VL teacher -> BAGEL student`：`zoo_scripts/zoo_run_opd_qwen2bagel_32b.sh`

启动脚本不再默认绑定本机模型路径。换机器后需要显式设置 BAGEL 代码仓库、student 模型和 teacher 模型路径。

## 1. 准备环境

先进入目标 verl-opd 仓库，安装或激活一个能跑 verl 的环境。这个环境至少需要包含 Ray、PyTorch、vLLM、Transformers、datasets、safetensors 以及 verl 的常规依赖。

示例：

```bash
cd /path/to/verl-opd
python -m pip install -e .
```

BAGEL 不是 HuggingFace `AutoModel` 架构，适配代码会在运行时 import 原始 BAGEL 仓库里的 `modeling/`、`data/` 等模块。因此必须准备一个本地 BAGEL repo，并设置：

```bash
export BAGEL_CODEBASE=/path/to/BAGEL
```

如果目标 Python 不在默认 `PATH` 里，可以设置：

```bash
export PYTHON_BIN=/path/to/verl-env/bin/python
```

## 2. 准备 MINT-CoT 数据

两个 OPD 启动脚本默认读取：

```text
projects/opd/data_parquet/mint_cot_dataset/train.parquet
projects/opd/data_parquet/mint_cot_dataset/val_sample100.parquet
```

在新机器上可以直接运行：

```bash
cd /path/to/verl-opd
PYTHON_BIN=${PYTHON_BIN:-python3} bash projects/opd/prepare_data/run_mint_cot_prep.sh
```

如果已经有原始 MINT-CoT parquet 文件：

```bash
RAW_MINT_FILE=/path/to/MINT-CoT_interleave_rl_54k_filtered.parquet \
PYTHON_BIN=${PYTHON_BIN:-python3} \
bash projects/opd/prepare_data/run_mint_cot_prep.sh
```

也可以让启动脚本读取外部已经准备好的 parquet：

```bash
export DATA_DIR=/path/to/mint_cot_dataset
export TRAIN_DATA_FILE=${DATA_DIR}/train.parquet
export VAL_DATA_FILE=${DATA_DIR}/val_sample100.parquet
```

## 3. 跑 BAGEL Teacher、Qwen Student

必须先设置：

```bash
export BAGEL_CODEBASE=/path/to/BAGEL
export STUDENT_MODEL=/path/or/hf-id/to/Qwen2.5-VL-7B-Instruct
export TEACHER_MODEL=/path/to/BAGEL-7B-MoT
```

启动：

```bash
cd /path/to/verl-opd
PYTHON_BIN=${PYTHON_BIN:-python3} \
bash zoo_scripts/zoo_run_opd_bagel2qwen_mintcot.sh 2 8
```

常用覆盖项：

```bash
TOTAL_STEPS=100 TRAIN_BATCH_SIZE=64 N_ROLL=2 \
PYTHON_BIN=${PYTHON_BIN:-python3} \
bash zoo_scripts/zoo_run_opd_bagel2qwen_mintcot.sh 2 8
```

这条路径中，student rollout 仍走 Qwen/vLLM，teacher ref log-prob 由 `BagelRefScorer` 计算。

## 4. 跑 Qwen Teacher、BAGEL Student

必须先设置：

```bash
export BAGEL_CODEBASE=/path/to/BAGEL
export STUDENT_MODEL=/path/to/BAGEL-7B-MoT
export TEACHER_MODEL=/path/or/hf-id/to/Qwen2.5-VL-32B-Instruct
```

启动：

```bash
cd /path/to/verl-opd
PYTHON_BIN=${PYTHON_BIN:-python3} \
bash zoo_scripts/zoo_run_opd_qwen2bagel_32b.sh 2 8
```

这条路径使用迁移后的 BAGEL FSDP engine 和 `bagel_single_turn_agent` rollout。BAGEL rollout 是 hybrid/colocated：它复用 BAGEL actor worker，不会再额外启动一个独立的 vLLM BAGEL 副本。

当前限制：外部 vLLM teacher scorer 还没有接进这条迁移后的默认路径。`USE_VLLM_TEACHER_REF=True` 会主动退出。默认配置使用 verl 标准 FSDP ref worker 来加载 Qwen teacher。

## 5. 转换 BAGEL Student Checkpoint

BAGEL student 训练出来的 verl checkpoint 是 FSDP shard，通常在：

```text
checkpoints/verl_opd/<experiment>/global_step_100/actor/
```

可以用下面的脚本转回 BAGEL checkpoint 目录：

```bash
${PYTHON_BIN:-python3} zoo_scripts/convert_verl_fsdp_to_bagel_hf.py \
  --actor-dir /path/to/checkpoints/verl_opd/<experiment>/global_step_100/actor \
  --base-bagel-dir /path/to/BAGEL-7B-MoT \
  --output-dir /path/to/output/BAGEL-OPD-step100 \
  --overwrite
```

这个转换脚本会做几件事：

- 读取 `--actor-dir/fsdp_config.json` 来确定 world size。
- 加载 `model_world_size_<N>_rank_<R>.pt` FSDP shard。
- 在 CPU 上合并 DTensor/FSDP shard。
- 复制 `--base-bagel-dir` 中 tokenizer、config 等非训练产物。
- 用合并后的 actor 权重替换输出目录里的 `ema.safetensors`。

建议先 dry-run 看 key 和 shape 是否正常：

```bash
${PYTHON_BIN:-python3} zoo_scripts/convert_verl_fsdp_to_bagel_hf.py \
  --actor-dir /path/to/global_step_100/actor \
  --base-bagel-dir /path/to/BAGEL-7B-MoT \
  --output-dir /tmp/bagel-opd-check \
  --dry-run
```

## 6. 换机器前检查清单

启动前确认：

- `BAGEL_CODEBASE` 指向原始 BAGEL repo，里面有 `modeling/`、`data/` 和 tokenizer/model 代码。
- `STUDENT_MODEL` 和 `TEACHER_MODEL` 已经按当前实验方向显式设置。
- MINT-CoT parquet 已经生成，或者 `run_mint_cot_prep.sh` 能读到/下载原始数据。
- Python 环境能 import `ray`、`torch`、`verl`、`datasets`、`safetensors` 和 BAGEL repo 模块。
- BAGEL student 路径下，`actor_rollout_ref.rollout.tensor_model_parallel_size` 需要覆盖所有 FSDP rank；当前脚本默认设置为 `NNODES * NGPU`。
- 当前 BAGEL OPD 路径只支持图片理解，不支持视频输入。

可以先跑这些静态检查：

```bash
bash -n zoo_scripts/zoo_run_opd_bagel2qwen_mintcot.sh
bash -n zoo_scripts/zoo_run_opd_qwen2bagel_32b.sh
python -m py_compile \
  projects/opd/prepare_data/mint_cot_dataset.py \
  zoo_scripts/convert_verl_fsdp_to_bagel_hf.py \
  verl/models/transformers/bagel_ref.py \
  verl/models/transformers/bagel_actor.py \
  verl/workers/engine/fsdp/bagel_impl.py \
  verl/workers/rollout/bagel_async_server.py \
  verl/experimental/agent_loop/bagel_single_turn_agent_loop.py
```
