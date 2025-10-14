
# Wheel-Legged Robot Reinforcement Learning Library

---

## 🧩 Introduction

This repository provides a **Reinforcement Learning (RL) framework** for wheel-legged robots.  
It supports **simulation-based policy training**, **Sim2Sim validation**, and **Sim2Real deployment** on real robots such as **Unitree Go2** and **custom wheel-legged platforms**.

### 🚀 Features
- ✅ Supports vision-proprioception fusion reinforcement learning  
- ✅ Compatible with IsaacGym, Gazebo, and PyBullet  
- ✅ End-to-end pipeline for training, validation, and deployment  
- ✅ Extensible to multiple robot platforms (Go2w, B2w, etc.)  
- ✅ Supports custom reward functions and curriculum learning

---

## ⚙️ Installation

### 1. Clone this repository
```bash
git clone https://github.com/XinLang2019/Wheel_Legged_Gym.git
cd Wheel_Legged_Gym
```

### 2. Create conda env
```bash
conda create -n wheel-legged python=3.8

conda activate wheel-legged
```

### 3. Install pytorch
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

### 4. Install Isaacgym
   - Download and install Isaac Gym Preview 3 (Preview 2 will not work!) from https://developer.nvidia.com/isaac-gym
   - `cd isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs `isaacgym/docs/index.html`)
   
### 5. Install rsl-rl
   -  `cd rsl_rl-1.0.2 && pip install -e .` 

### 6. Install legged-gym
```bash
cd legged_gym && pip install -e .
```

### 7.Install Wheel-Legged-Gym
```bash
cd .. 
pip install -e .

```

## Use

### 1. train 
```bash
cd legged_gym
python legged_gym/scripts/train.py --task=go2w --headless
```

if you need train in different GPU :
```bash
python legged_gym/scripts/train.py --task=b2w --headless --sim_device=cuda:0 --rl_device=cuda:0
```

### 2. play policy
```bash
cd legged_gym
python legged_gym/scripts/play.py --task=go2w --num_envs=20
```


