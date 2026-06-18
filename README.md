# GCNGrasp-VP: Affordance-Guided View Planning for Efficient Task-Oriented Grasping

[![arXiv](https://img.shields.io/badge/arXiv-2606.19091-red?style=flat&logo=arxiv)](https://arxiv.org/abs/2606.19091)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface)](https://huggingface.co/TongZJ/GCNGrasp-v2)

🎉 **Congratulations! This paper has been accepted by IROS 2026!**

> Task-oriented grasping performance degrades significantly when object views suffer from occlusions. Existing task-oriented grasping methods typically assume task-relevant regions are visible in the initial frame, while view planning approaches enable active perception but often ignore task semantics and rely on time-consuming scene reconstruction. To address these limitations, we present GCNGrasp-VP, an efficient framework integrating affordance field prediction with active view planning. Central to this framework is GCNGrasp-v2, a task-oriented grasp model that simultaneously supports grasp evaluation and affordance field prediction, achieving constant-time inference complexity. Leveraging this capability, our Affordance-guided View Planner (Affordance-VP) utilizes the affordance field as an information gain metric to guide camera observation of task-relevant regions without requiring scene reconstruction. View planning results show that our method significantly outperforms scene-uncertainty-driven baselines with only one view adjustment. Real-world validation further confirms substantial improvements in grasp success rates for single-object scenarios while maintaining millisecond-level computational latency.

## 🔧 Environment Setup

```bash
conda create -n gcngraspvp python=3.10
conda activate gcngraspvp

pip install "numpy<2" torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r gcngrasp/requirements.txt -v --no-build-isolation

# For demos
pip install -r gcngrasp/system/requirements.txt
```

---

## 🚀 Usage

### 🎯 Training (`main.py`)

Trains on [TaskGrasp](https://github.com/adithyamurali/TaskGrasp) dataset.

```bash
python main.py --fold <fold> --train <version> [--devices 0 1]
python main.py --resume <path>
python main.py --test <path> [--test-bs-div 4]
```

**Parameters:**
- `--train`: Model version (`1`, `2`, `2aff`, `gpt`)
- `--fold`: Data fold (`o0`, `o1`, `t0`, `t1`)
- `--resume`/`--test`: Checkpoint path

---

### 🎮 Demo (`demo.py`)

Modify the function call at the bottom of the file:

```python
inference_dataset(obj_id=20, task_id="pour")  # Test on TaskGrasp database
inference_img(CONFIG, *virtual_robot.frame_from_folder(ASSETS[1]))  # Test on assets folder
```

> **Note:** The best model has been uploaded to HuggingFace ([TongZJ/GCNGrasp-v2](https://huggingface.co/TongZJ/GCNGrasp-v2)). When running the demo, it will be automatically downloaded.
>
> **Tip:** If you don't want to manually look up WordNet synsets (e.g., `frying_pan.n.01`) or TaskGrasp task categories, you can use `InstructionConverter` to convert natural language instructions directly:
> ```python
> from gcngrasp.utils.instruct_cvt import InstructionConverter
> ins_cvt = InstructionConverter("qwen3.5-flash-2026-02-23")
> result = ins_cvt("use the pan to pour")  # {'cls': 'frying_pan.n.01', 'task': 'pour'}
> ```

---

### 📷 NBV Demo (`demo_nbv.py`)

Offline NBV demonstration on assets folder.

**Assets naming:** `{camera}--{target_obj}--{obj_category}--{task}[--{variant}]`

Example: `RS-D405--green-pot--frying_pan.n.01--pour--0`

---

### 🤖 Real Robot (`real_nbv.ipynb`)

Online NBV with robot arm. Same as `demo_nbv.py`, but integrates robot control via ZMQ:

```python
robot_ctrl = my_robot.get_robot_client(addr="<robot_ip>:5555")
# ... integrate frame, plan NBV ...
robot_ctrl.move(Teb)
```

Customize by modifying `my_robot.py` to implement `get_frame()`, `move()`, `grip()`.

> **Important:** Robot configurations and control interfaces vary significantly across platforms. Before using this script with a real robot, carefully review the configuration files, variables, and functions used in `real_nbv.ipynb` to ensure compatibility with your specific hardware setup.

---

## 📝 Other Scripts

### `debug.py`

A draft / scratchpad for internal feature validation. Contains miscellaneous examples and experiments that may be useful for reference, but is not maintained as a polished tool.

### `main_nbv.py`

Evaluates the NBV (Next Best View) planning pipeline. 

> **Note:** We do not release the NBV evaluation dataset publicly because we do not want this small-scale dataset to inadvertently become a standard benchmark for validating the model. We encourage researchers to build larger, more compelling datasets for evaluation.

### `optim_nbv_params.py`

Bayesian optimization script for tuning NBV planner parameters (e.g., `occ`, `elev` loss weights).

---

## 📦 Downloads

Model weights and training logs are released via [OneDrive](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvYy85NWY4MDYyMjk3Njc0NDIwL0lnQjJfUWt3Q0dOLVJhMHM4WHo5WE5ha0FSU0MtSk9lV1U0ZTJEa3BVb0xKelRJP2U9akQ2M3ZM&id=95F8062297674420%21s3009fd766308457ead2cf17cfd5cd6a4&cid=95F8062297674420&sb=name&sd=1):
- `runs/train/`: GCNGrasp-v2 model weights and training logs for two branches across 8 experimental settings
