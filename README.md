# GCNGrasp-VP: Affordance-Guided View Planning for Efficient Task-Oriented Grasping

## 🔧 Environment Setup

```bash
conda create -n gcngraspvp python=3.10
conda activate gcngraspvp

pip install numpy<2 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
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
inference_dataset()  # Test on TaskGrasp database
inference_img()      # Test on assets folder
```

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

---

## 📦 Downloads

Model weights released via cloud drive:
- `checkpoints/` — NBV inference models
- `runs/train/` — Training checkpoints
