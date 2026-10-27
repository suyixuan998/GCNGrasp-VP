# GCNGrasp-VP 本机环境安装与使用教程

本文记录 `/home/share/GCNGrasp-VP` 在本机已经验证的安装方案，环境名为 **gcngraspvp**。最后核对日期：**2026-10-27**。

- 已有环境，直接运行 demo：看第 1 节。
- 新机器从零配置：按第 3～8 节顺序安装，再看第 9～11 节准备权重和验证。
- 数据集训练：看第 13 节；当前训练使用本地适配加载器，与作者训练协议存在区别。
- 依赖警告、路径问题和显存问题：看第 12 节。

官方参考：[GCNGrasp-VP](https://github.com/Instinct323/GCNGrasp-VP)、[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)、[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)、[SAM2](https://github.com/facebookresearch/sam2)、[gsplat](https://github.com/nerfstudio-project/gsplat)。

## 1. 已有环境：直接运行

本机已经安装好 GCNGrasp、Contact-GraspNet、GroundingDINO、SAM2、DA3、xformers 和 gsplat，单帧 NBV demo 所需权重也已准备。

```bash
source /home/suyixuan/anaconda3/etc/profile.d/conda.sh
conda activate gcngraspvp
cd /home/share/GCNGrasp-VP

python demo_nbv.py
```

运行前可以检查 Python 和 GPU：

```bash
which python
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

本机 Python 应为 `/home/share/suyixuan/Anaconda3/envs/gcngraspvp/bin/python`，CUDA 可用性应输出 `True`。

程序依次执行目标检测与分割、深度映射、抓取生成、任务抓取评分、NBV 规划，并打开 Open3D 窗口。终端出现 `Action(..., grasp=False)` 表示这次选择了调整观察视角；关闭可视化窗口后，该单帧 demo 正常退出。

当前入口使用 `ASSETS[1]`。`Path.iterdir()` 没有排序保证，本机此前对应刷子样例，其他机器上的第二个目录可能不同。可查看实际顺序：

```bash
python -c "from pathlib import Path; print([(i, p.name) for i, p in enumerate(Path('assets').iterdir())])"
```

## 2. 本机环境与关键版本

| 项目 | 已核对的配置 |
|---|---|
| 系统 | Ubuntu 22.04.5 LTS，x86_64 |
| GPU | NVIDIA GeForce RTX 4090 D，计算能力 8.9 |
| Conda 初始化脚本 | `/home/suyixuan/anaconda3/etc/profile.d/conda.sh` |
| 环境目录 | `/home/share/suyixuan/Anaconda3/envs/gcngraspvp` |
| Python | 3.10.22 |
| CUDA Toolkit / nvcc | 11.8，`/usr/local/cuda-11.8` |
| PyTorch | 2.5.1+cu118 |
| torchvision / torchaudio | 0.20.1+cu118 / 2.5.1+cu118 |
| NumPy / OpenCV / Open3D | 1.26.4 / 4.11.0.86 / 0.20.0 |
| torch-cluster / torch-geometric | 1.6.3 / 2.8.1 |
| Transformers / huggingface-hub | 4.40.2 / 0.36.2 |
| timm / supervision | 1.0.30 / 0.31.2 |
| Dash / Pydantic | 3.4.0 / 1.10.26 |
| Lightning / pandas | 2.6.6 / 2.3.3 |
| xformers / gsplat | 0.0.29 / 1.5.2 |

`nvidia-smi` 中的 “CUDA Version” 表示驱动支持的 CUDA 版本上限，不是当前安装的 Toolkit 或 PyTorch 的 CUDA 版本。编译扩展时重点核对 `nvcc --version` 与 `torch.version.cuda`；本机两者均为 11.8。

本地兼容性约束保存在 [config/requirements-local-constraints.txt](config/requirements-local-constraints.txt)。它是 **pip constraints 文件**，用于限制版本，不能代替完整依赖列表。

官方 README 使用 PyTorch **cu124**。本机已有 Toolkit 11.8，因此实际采用 **cu118**；PyTorch 主版本仍为官方要求的 2.5.1。下文完整安装命令统一使用 cu118，不要在同一环境中交叉执行 cu118 与 cu124 的安装步骤。

## 3. 从零准备系统、项目和 Conda 环境

已有项目、已有环境时跳过相应创建步骤。下面的绝对路径针对本机；换电脑需要替换 Conda 初始化脚本和项目位置。

### 3.1 系统工具

先检查工具是否已经存在：

```bash
git --version
g++ --version
nvcc --version
nvidia-smi
```

在新的 Ubuntu 机器上，缺少编译和显示基础库时可安装：

```bash
sudo apt-get update
sudo apt-get install -y build-essential git cmake libgl1 libglib2.0-0
```

CUDA Toolkit 需要与所选 PyTorch CUDA 构建匹配。本机已安装 `/usr/local/cuda-11.8`，无需重新安装驱动或 Toolkit。

### 3.2 获取项目、创建环境

```bash
cd /home/share
git clone https://github.com/Instinct323/GCNGrasp-VP.git
cd /home/share/GCNGrasp-VP

source /home/suyixuan/anaconda3/etc/profile.d/conda.sh
conda create -n gcngraspvp python=3.10 -y
conda activate gcngraspvp
```

本教程还使用了当前工作区新增的约束文件、修补脚本和测试文件。**仅 clone 官方仓库不会包含这些本地文件**，迁移时需一并复制：

- `config/requirements-local-constraints.txt`
- `scripts/fix_groundingdino_warnings.py`
- 如需执行本地回归测试，再复制 `tests/` 和相关本地适配文件。

后续始终在项目根目录、激活的环境中使用 `python -m pip`，以确保包装入正确环境：

```bash
which python
python -m pip --version
```

### 3.3 编译环境

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH"
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST="8.9"

python -m pip install "setuptools==84.0.0" wheel "ninja==1.13.2"
```

`MAX_JOBS` 限制编译并发数；内存不足时可以降低到 2。`8.9` 对应本机 RTX 4090 D；其他 GPU 应使用对应计算能力，或不设置 `TORCH_CUDA_ARCH_LIST` 让 PyTorch 自动选择。

## 4. 安装 PyTorch 和项目基础依赖

先安装 NumPy，再从 PyTorch 官方 cu118 索引安装匹配的一组包：

```bash
python -m pip install -c config/requirements-local-constraints.txt "numpy==1.26.4"

python -m pip install -c config/requirements-local-constraints.txt \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu118
```

先确认 GPU 可用，再安装需要链接 PyTorch 的扩展：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

python -m pip install -c config/requirements-local-constraints.txt \
  -r gcngrasp/requirements.txt --no-build-isolation -v
```

`--no-build-isolation` 让 `torch-cluster` 使用当前环境已经安装好的 PyTorch 编译，避免临时构建环境另装不匹配的 PyTorch。本机的 `torch-cluster` 已成功编译。

## 5. 安装 demo 依赖、GroundingDINO 和 SAM2

### 5.1 通用 demo 依赖

```bash
python -m pip install -c config/requirements-local-constraints.txt \
  colorama logdir pyrealsense2 pyzmq supervision provider pyrender \
  typeguard socksio dash transformers huggingface-hub timm
```

这里保留官方 `gcngrasp/system/requirements.txt` 中的通用包，并补上本机实际需要的兼容性依赖。离线 demo 使用 `assets/`，不要求接入 RealSense 相机或机械臂。

### 5.2 从官方 Git 仓库安装

为与本机已测试版本一致，固定源码提交：

```bash
python -m pip install --no-build-isolation \
  -c config/requirements-local-constraints.txt \
  git+https://github.com/IDEA-Research/GroundingDINO.git@856dde20aee659246248e20734ef9ba5214f5e44

python -m pip install --no-build-isolation \
  -c config/requirements-local-constraints.txt \
  git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4
```

检查 CUDA 扩展能否导入：

```bash
python -c "import torch; from groundingdino import _C; from sam2 import _C as sam_C; print('GroundingDINO / SAM2 CUDA 扩展导入成功')"
```

SAM2 某些构建失败会允许安装继续，因此不能只看 pip 最后的 “Successfully installed”，还要执行上述扩展检查。

## 6. 安装 Depth Anything 3 和 xformers

本机使用与 PyTorch 2.5.1+cu118 匹配的官方 xformers wheel：

```bash
python -m pip install --no-deps xformers==0.0.29 \
  --index-url https://download.pytorch.org/whl/cu118

python -m pip install --no-deps \
  git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@3d835ec1a5802d64a8b8b15f817a1ab54809bfe4

python -m pip install -c config/requirements-local-constraints.txt depth-anything-3
```

第二步按 GCNGrasp-VP 的说明先从 Git 安装 DA3 源码，第三步再为这个已经安装的包补齐声明的依赖。约束文件会固定 MoviePy 1.0.3、FastAPI 0.119.1 等兼容版本。

已安装的主要附加依赖包括 `einops`、`e3nn`、`evo`、`moviepy`、`pillow-heif`、`plyfile`、`pycolmap`、`fastapi`、`uvicorn` 等。

`conda list` 显示 `depth-anything-3 0.0.0 pypi_0 pypi` 是正常的：`0.0.0` 是官方项目声明的 Python 包版本，V3 表示 Depth Anything 的第三代；`pypi_0 / pypi` 是 Conda 对 pip 安装包的标记，源码实际来自上面的 GitHub 提交。

## 7. 安装 gsplat，启用 3DGS 渲染依赖

DA3 官方要求的 gsplat 提交为 `0b4dddf04cb687367602c01196913cde6a743d70`，其包版本为 1.5.2：

```bash
python -m pip install -c config/requirements-local-constraints.txt ninja jaxtyping

python -m pip install --no-build-isolation \
  -c config/requirements-local-constraints.txt \
  git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
```

本机编译产物为：

```text
/home/share/suyixuan/Anaconda3/envs/gcngraspvp/lib/python3.10/site-packages/gsplat/csrc.so
```

gsplat 已通过 CUDA RGB/深度渲染、反向传播以及 DA3 `render_3dgs()` 接口验证。当前 `DA3MONO-LARGE` 用于单目深度估计；安装渲染依赖本身不会使这个模型具备未配置的 Gaussian 预测头。

## 8. 应用已验证的警告修复

### 8.1 修复内容

官方 GroundingDINO 0.1.0 的部分 Python 调用与本环境依赖版本之间存在弃用警告。执行本地修补脚本：

```bash
python scripts/fix_groundingdino_warnings.py --check
python scripts/fix_groundingdino_warnings.py
```

脚本针对已安装包修改 7 个 Python 文件，重新执行会检查并跳过已经修补的内容：

| 原警告 | 实际修复 |
|---|---|
| `timm.models.layers` 弃用 | 使用 `timm.layers` 导入 |
| `torch.meshgrid` 缺少 indexing | 明确设置 `indexing="ij"`，保留原索引方式 |
| `torch.load` 未指定 weights_only | 使用 `weights_only=True`；当前官方 SwinB 权重已验证可加载 |
| BERT 的 device 参数弃用 | 移除注意力掩码接口中已弃用的 device 参数 |
| checkpoint 缺少 use_reentrant | 明确使用 `use_reentrant=False` |
| 推理时 checkpoint 输入无梯度 | 仅在训练且启用梯度时执行 activation checkpoint |
| 旧 AMP autocast 接口 | 使用 `torch.amp.autocast("cuda", ...)` |

本项目 `gcngrasp/api/utils/sv_extension.py` 也已修复：先筛选有限的中心点与邻点，再做距离相减，避免无效深度或填充点造成 `inf - inf`。有效邻点的计数规则保持不变。

这些修改解决触发原因，没有添加全局警告屏蔽。`demo_nbv.py` 入口文件保持官方原样；由于数值处理函数和已安装依赖已有兼容性修补，不能把整个运行栈称作完全未修改的官方源码。

### 8.2 备份和回退

默认备份、文件哈希和完整补丁保存在：

```text
runs/groundingdino_warning_patch/
├── manifest.json
├── changes.patch
└── originals/
```

查看是否需要修补、恢复依赖原文件：

```bash
python scripts/fix_groundingdino_warnings.py --check
python scripts/fix_groundingdino_warnings.py --restore
```

`--restore` 只恢复该脚本修改的 GroundingDINO 文件，不回退项目的掩码数值修复。如果依赖文件后来又被手动修改，脚本会拒绝直接覆盖。

重新安装 GroundingDINO 会覆盖环境里的 Python 修补；安装后需重新运行修补脚本。`pip freeze` 也不会保存这种文件级修改。

## 9. 下载权重、理解缓存与项目 weights 目录

### 9.1 代码包与权重是两类文件

| 模型 | 作用 | 本机包版本 |
|---|---|---|
| Contact-GraspNet | 生成几何抓取候选 | 代码随本项目放在 `gcngrasp/api/third_party/` |
| GroundingDINO | 用文本描述检测目标 | groundingdino 0.1.0 |
| SAM2 | 根据检测框分割目标 | SAM-2 1.0 |
| BERT | GroundingDINO 的文本编码器 | 通过 Transformers 加载 |
| GCNGrasp-v2 | 按物体类别和任务评估抓取，预测 affordance | 代码在本项目中 |
| DA3 | 深度估计与映射 | depth-anything-3 0.0.0 |

`pip install git+...` 会临时克隆 Git 仓库，将 Python 源码和编译扩展安装到环境，然后通常删除临时克隆目录。正常调用不需要另外保留独立 Git 仓库；要修改上游源码时可以自行 clone 并使用 editable 安装。

本机这些包的源码在：

```text
/home/share/suyixuan/Anaconda3/envs/gcngraspvp/lib/python3.10/site-packages/
├── groundingdino/
├── sam2/
├── depth_anything_3/
└── gsplat/
```

### 9.2 新机器建议统一缓存位置

项目的 HuggingFace 路径查找硬编码使用 `~/.cache/huggingface/hub`。新环境可以统一设置：

```bash
export HF_HOME="$HOME/.cache/huggingface"
export TORCH_HOME="$HOME/.cache/torch"
export HF_HUB_DISABLE_XET=1
```

这里的变量只影响当前终端及其子进程。已有共享缓存的本机不必重新设置或重复下载，现有链接关系见第 9.4 节。

先下载 HuggingFace 模型。以下命令显式使用官方站点；网络访问困难时可将命令中的 `HF_ENDPOINT` 换成 `https://hf-mirror.com`：

```bash
HF_ENDPOINT=https://huggingface.co hf download TongZJ/GCNGrasp-v2

HF_ENDPOINT=https://huggingface.co hf download bert-base-uncased \
  config.json tokenizer_config.json tokenizer.json vocab.txt pytorch_model.bin

HF_ENDPOINT=https://huggingface.co hf download depth-anything/DA3MONO-LARGE \
  --include '*.json' '*.safetensors'
```

单帧 `demo_nbv.py` 使用 `DA3MONO-LARGE`，约 1.27 GiB。原代码在多帧模式还可能加载 `DA3-GIANT-1.1`；该权重尚未在本机下载，当前单帧 demo 不需要它。

项目 `gcngrasp/api/utils/__init__.py` 在导入时会将 `HF_ENDPOINT` 设为镜像地址；上述独立 CLI 命令先准备本地缓存后，运行 demo 可直接读取这些文件。

### 9.3 下载三个 Torch 模型权重

下面调用项目自带下载函数，目标是当前 `torch.hub.get_dir()` 下的 `checkpoints/`；已有文件会复用：

```bash
python - <<'PY'
from gcngrasp.api.utils import download_checkpoint

models = [
    ('https://github.com/elchun/contact_graspnet_pytorch/raw/main/checkpoints/contact_graspnet/checkpoints/model.pt', 'contact_graspnet.pt'),
    ('https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth', None),
    ('https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt', None),
]
for url, name in models:
    print(download_checkpoint(url, filename=name))
PY
```

本项目的 `.pt` 来自其指定的 Contact-GraspNet PyTorch 实现，与 NVIDIA 原始 TensorFlow 实现的 checkpoint 格式不同，不应直接互换。

### 9.4 本机已经保存的文件

项目 `weights/` 中是独立保存的模型文件：

```text
/home/share/GCNGrasp-VP/weights/
├── contact_graspnet.pt                      # 25.4 MiB
├── groundingdino_swinb_cogcoor.pth           # 894.6 MiB
├── sam2.1_hiera_large.pt                    # 856.5 MiB
└── GCNGrasp-v2/
    ├── model_v2aff.yaml
    ├── train_v2aff.yaml
    ├── hparams.yaml
    └── checkpoints/best.ckpt                # 136.7 MiB
```

三个 Torch 模型的原缓存保留在 `/home/suyixuan/.cache/torch/hub/checkpoints/`。本机当前 `TORCH_HOME=/home/share/suyixuan/cache/torch`，其 `hub/checkpoints/` 下同名文件通过软链接指向原缓存。

| 模型 | 程序读取的模型目录或文件 |
|---|---|
| GCNGrasp-v2 | `/home/suyixuan/.cache/huggingface/hub/models--TongZJ--GCNGrasp-v2/snapshots/3fefe9a977a3b566c086a3f75fbea406f0eedaee/checkpoints/best.ckpt` |
| BERT | `/home/share/suyixuan/cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594/pytorch_model.bin` |
| DA3MONO-LARGE | `/home/share/suyixuan/cache/huggingface/hub/models--depth-anything--DA3MONO-LARGE/snapshots/f465978e618db8cc79c83b8bbf24964857db1875/model.safetensors` |

BERT 和 DA3 在项目硬编码的 `~/.cache/huggingface/hub/` 下有对应模型目录软链接，指向上面的共享缓存。HuggingFace 的 `snapshots/` 文件本身通常也会链接到 `blobs/`，属于正常缓存结构。

**复制文件到 `weights/` 不会自动改变程序的加载路径。**当前 `config/system/params.yaml` 仍指定 `tog_model.path: TongZJ/GCNGrasp-v2`；三个 Torch 模型仍使用缓存。BERT 和 DA3 权重尚未复制到项目 `weights/`。

## 10. 安装完成后的验证

### 10.1 包依赖和导入

```bash
python -m pip check

python - <<'PY'
import torch
from groundingdino import _C
from sam2 import _C as sam_cuda
from depth_anything_3.api import DepthAnything3
from gsplat import rasterization
print('关键模型及 CUDA 扩展导入成功')
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
PY
```

预期 `pip check` 输出 `No broken requirements found.`，DA3 导入时不再出现缺少 gsplat 的提示。

### 10.2 xformers 实际 CUDA 运算

```bash
python - <<'PY'
import torch
from xformers.ops import memory_efficient_attention
q = torch.randn(1, 64, 4, 32, device='cuda', dtype=torch.float16)
y = memory_efficient_attention(q, q, q)
torch.cuda.synchronize()
assert torch.isfinite(y).all()
print('xformers CUDA attention:', tuple(y.shape))
PY
```

### 10.3 本地回归测试与严格警告检查

```bash
OMP_NUM_THREADS=2 python -m unittest discover -s tests -v

python -W error::FutureWarning -W error::UserWarning -W error::RuntimeWarning demo_nbv.py
```

第二条会将这三类警告提升为异常，适合检查修复效果。正常日常使用仍可运行 `python demo_nbv.py`。

已验证的 gsplat CUDA 渲染及反向传播日志见 `runs/demo_nbv_20261027/gsplat_validation.log`，其来源和环境信息保存在 `result_gsplat.json`。

## 11. Demo 输入、运行结果和代码修改范围

离线样例目录命名遵循：

```text
{camera}--{target_obj}--{obj_category}--{task}[--{variant}]
```

例如 `RS-D405--green-pot--frying_pan.n.01--pour--0`。目录包含 `color.png`、`depth.png`，相机参数从 `config/system/RS-D405.yaml` 读取。换成自己的数据时，需匹配相机内参、图像尺寸与深度缩放单位。

本机离线示例无需完整 TaskGrasp 数据集。`demo_nbv.py` 的入口、网络结构、NBV 规划代码和 `config/system/` 保持原有内容。本次为消除警告，修改了项目的 `sv_extension.py` 与环境里的 GroundingDINO Python 文件；另有第 13 节所述本地训练适配改动。

`[INFO] using MLP layer as FFN`、`Loading weights from local directory` 和 `Timer` 都是正常信息。`scene/mapping` 表示 DA3 映射已执行；`planner/solve` 表示视角规划耗时。它们是本次模块计时，不能直接作为跨机器性能对比。

本次警告修复后，严格警告检查下的 demo 输出了 `Action(..., grasp=False)` 并正常退出。NBV 动作矩阵与此前 DA3 测试记录一致。

## 12. 常见问题

### 12.1 Open3D 导入时找不到 GetCoreSchemaHandler

原因是 Dash 4 要求新版 Pydantic，而官方 demo 依赖中的 `provider` 要求 Pydantic 1。本机采用 Dash 3.4.0、Pydantic 1.10.26：

```bash
python -m pip install -c config/requirements-local-constraints.txt dash pydantic
python -m pip check
```

### 12.2 Transformers 禁用了 PyTorch

本次安装曾选中要求 PyTorch >= 2.6 的 Transformers 5，与项目 PyTorch 2.5.1 不匹配。使用已验证组合：

```bash
python -m pip install -c config/requirements-local-constraints.txt transformers huggingface-hub
```

### 12.3 xformers 或 CUDA 扩展不能加载

先核对：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvcc --version
which python
which ninja
```

xformers wheel 需要与 Python、PyTorch 和 CUDA 构建匹配；本机应使用第 6 节的 cu118 wheel。GroundingDINO、SAM2 和 gsplat 应在安装好 PyTorch 后，使用同一环境和 Toolkit 编译。迁移到不同 GPU 架构后可能需要重新编译。

### 12.4 HuggingFace 已下载，但项目仍找不到文件

检查实际缓存与项目硬编码路径：

```bash
python - <<'PY'
import os
from pathlib import Path
import torch
print('HF_HOME:', os.environ.get('HF_HOME'))
print('TORCH_HOME:', os.environ.get('TORCH_HOME'))
print('Torch Hub:', torch.hub.get_dir())
print('项目查找的 HF 目录:', Path('~/.cache/huggingface/hub').expanduser())
PY
```

使用其他 `HF_HOME` 时，需要将对应模型目录链接或保存到项目查找位置。本机已有 BERT/DA3 链接。项目辅助函数仅检查快照目录是否存在，下载未完成也可能留下目录；这种情况应重新执行第 9 节对应下载命令，确认权重文件完整。

SOCKS 代理提示缺少 `socksio` 时，使用当前环境安装该包；不要把代理凭证写入项目教程或日志。

### 12.5 显存不足或窗口没有退出

查看 `nvidia-smi`。Open3D 窗口打开期间 demo 进程通常仍保留模型显存，关闭该窗口后才释放。已有训练或其他推理任务时，需要为本次 demo 留出足够显存，避免重复启动同一个 demo。

`draw_geometries()` 需要可用图形会话，检查 `echo "$DISPLAY"`；当前本机显示为 `:1`。纯 SSH、无显示会话的机器需要另行配置显示或离屏可视化。

### 12.6 重新安装后，警告又出现

重新运行第 8 节修补脚本。不要通过全局屏蔽所有警告替代兼容性修复；本机回归测试覆盖有效点、无效深度、边界填充、BERT 输出、推理 checkpoint 分支与训练梯度。

## 13. 本地 TaskGrasp 训练与严格复现边界

### 13.1 数据与本地加载器

本机数据根目录为 `/home/share/GraspGPT_public/data/taskgrasp`，写在 `config/data.yaml`。当前仓库原有导入需要 `gcngrasp.dataset`，本机新增了 `gcngrasp/dataset/task_grasp.py` 适配该数据。

需要的数据内容包括 `scans/`、`splits_final/`、`task1_results.txt`、`misc.pkl`。加载器使用官方划分文件，`o0` 对应 `splits_final/o/0` 的类别留出；实例留出可用 `i0`，任务留出可用 `t0`。

本地实现的具体处理：

- 完整点云读取 `fused_pc_clean.npy`，减去原始完整点云均值，使其与标注抓取处于同一坐标系。
- 从 TaskGrasp 原始七点夹爪折线去重、排序，形成模型要求的六个控制点。
- 缺少作者的部分视图预处理，采用 Open3D hidden-point-removal 生成 8 个上半球模拟视图，缓存在 `runs/taskgrasp_cache/`。
- Affordance 目标按论文公式 (10) 由当前划分的同物体、同任务抓取计算；无正样本时不产生该目标的梯度；目标、点云和抓取同步随机旋转。
- 保留原始重复和冲突的 `handover` 标注，检查划分间标注与留出单位是否重叠。
- 分类指标使用 0.5 阈值，mAP 在第一阶段 True/Weak True 的物体任务对上计算。

**这是本地训练适配版。模拟部分视图和评估实现没有与作者未提供的模块逐项对齐，不能将训练结果直接当作论文结果的严格复现。**

### 13.2 配置、启动和查看状态

`config/train_v2aff_local.yaml`：200 epochs、batch size 8、梯度累积 8，每个优化步有效 batch size 64；4096 点、每组 25 个抓取、AdamW、初始学习率 0.001。小 batch 的 BatchNorm 统计不等同于原版 batch size 64。

确认没有重复任务后，训练命令为：

```bash
python -u main.py --fold o0 --train 2aff --devices 0 \
  --train-config config/train_v2aff_local.yaml --skip-cuda-wait
```

`--skip-cuda-wait` 允许与现有 Python CUDA 进程共用 GPU，不预留显存。没有该参数时，原入口会等待其他匹配的 Python CUDA 进程退出。

本机已有训练记录，日志和进程信息可查看：

```bash
cat runs/2aff_o0_local/process.json
tail -f runs/2aff_o0_local/train.log
tensorboard --logdir runs/2aff_o0_local
```

以 `process.json` 的 PID、当前进程及日志为准，本文不将某次运行状态永久写成“仍在训练”。已有记录的实验目录为 `runs/2aff_o0_local/lightning_logs/version_0`。

```bash
# 中断后恢复
python main.py --resume runs/2aff_o0_local/lightning_logs/version_0 \
  --devices 0 --skip-cuda-wait

# 完整训练后测试最优权重
python main.py --test runs/2aff_o0_local/lightning_logs/version_0 \
  --test-bs-div 4 --skip-cuda-wait
```

`checkpoints/best.ckpt` 与 `checkpoints/last.ckpt` 保存最优和最近权重。实验目录同时保存模型、训练、数据 YAML，恢复或测试优先读取该实验的数据配置。

此前的训练冒烟记录：训练/验证/测试分别包含 167,700 / 36,650 / 63,600 条原始标注；真实数据两步 GPU 训练和验证通过，冒烟峰值分配显存约 2.34 GiB。这些是历史验证记录，不代表完整 200 轮训练已经完成。

## 14. 本地改动、环境迁移与验证记录

本地训练修改涉及 `.gitignore`、`config/data.yaml`、`gcngrasp/__init__.py`、`gcngrasp/requirements.txt`、`gcngrasp/utils/analysis.py`、`gcngrasp/utils/lit_extension.py`、`main.py`，并新增数据加载器、训练配置和回归测试。警告修复另外修改了 `gcngrasp/api/utils/sv_extension.py`，增加 GroundingDINO 修补脚本和兼容性测试。

可以自行核对：

```bash
git status --short
git diff --stat
git diff -- demo_nbv.py
```

`demo_nbv.py` 本次核对的 SHA-256 为：

```text
cef5d4da7f1c5b2c7b62c24e812901e7ef1282c3525d5f13a2681c0824ea3014
```

迁移时保存环境清单：

```bash
mkdir -p runs/env_export
python -m pip freeze > runs/env_export/requirements-freeze.txt
conda env export --no-builds > runs/env_export/environment.yml
```

导出文件可能包含本机绝对路径，跨机器重建时优先使用本文的分步安装及约束文件；重建后重新执行依赖修补和 CUDA 验证。下载权重、训练数据、缓存和自定义代码需要另外保留。

| 验证记录 | 文件 |
|---|---|
| 最初的 RGB-D demo（未安装 DA3 时） | `runs/demo_nbv_20261027/demo_nbv.log`、`result.json`、`visualization.png` |
| 安装 DA3 后的 demo | `runs/demo_nbv_20261027/demo_nbv_da3.log`、`result_da3.json`、`visualization_da3.png` |
| gsplat 编译、渲染、梯度验证 | `runs/demo_nbv_20261027/gsplat_install.log`、`gsplat_validation.log`、`result_gsplat.json` |
| 本次无警告 demo | `runs/demo_nbv_20261027/demo_nbv_no_warnings.log`、`result_no_warnings.json` |
| GroundingDINO 文件修补与原件 | `runs/groundingdino_warning_patch/` |
| 教程整理前的备份 | `runs/documentation_backup/ENV_SETUP_LOCAL.before_rewrite.md` |

`runs/` 和项目 `weights/` 目录被 Git 忽略，权重单独保存到 Hugging Face；这些本地记录不会自动随普通源码提交或 clone 迁移。
