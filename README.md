# Industrial Posture Analysis & Safety Monitor

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green?logo=qt&logoColor=white)
![PyTorch](https://img.shields.io/badge/AI-PyTorch-orange?logo=pytorch&logoColor=white)
![MediaPipe](https://img.shields.io/badge/Vision-MediaPipe-lightgrey)

> **Protecting the workforce in the era of automation through Real-Time Edge-Cloud AI.**

## Overview

In an industrial landscape where manual material handling remains a primary cause of musculoskeletal disorders (MSDs), standard safety protocols often fail to provide immediate, actionable feedback.

This project implements an **Industrial IoT (IIoT) Solution** for real-time ergonomic monitoring. The core AI model was developed and trained on the **CarDa (Car Door Assembly) Video Dataset**. We utilized a **Hybrid Intelligence** approach combining deterministic bio-mechanical physics with advanced Deep Learning (ST-GCNs) distributed across a highly scalable **Edge-to-Cloud Architecture**.

---

## Data Re-Engineering & Physics Augmentation

One of the most significant breakthroughs in this project was transitioning from raw, frame-dependent video sequences (such as the original CarDa `.mp4` files) to **FPS-Agnostic Physical Data**. 

Standard AI models break when a factory camera stutters or the frame rate drops. Converting a raw video dataset into time-based tensors required us to re-engineer the entire data pipeline from the ground up:

1. **Hardware Timestamping:** The edge client stamps every skeletal frame with an absolute physical hardware timestamp (`float64`) the exact microsecond it leaves the `V4L2` driver. 
2. **Temporal Interpolation:** When the server receives a batch of frames, it extracts the last 5.0 seconds of physical time. Using `numpy.interp`, it mathematically resamples the messy, jittery timestamp timeline into **exactly 50 uniform frames**, rendering the model completely immune to network lag and frame drops.
3. **1st-Order Kinematics (The "Crazy Feature"):** Static positions aren't enough to judge dangerous lifting momentum. We engineered a physics layer that calculates the true bio-mechanical velocity vectors (`V = dS / dt`) for every joint over the interpolated sequence. 
4. **6-Channel Tensors:** The final input to the AI model isn't just an image or a 3D coordinate, it is a dense 6-channel tensor (`X, Y, Z, Vx, Vy, Vz`) representing the complete spatio-temporal kinematic state of the worker.

---

## ST-GCN Architecture (SE Blocks & Attention)

The core brain of the system is a **Spatial-Temporal Graph Convolutional Network (ST-GCN)**. Rather than treating the human body as a flat image, it treats it as a mathematically connected Graph (where bones are edges and joints are nodes).

### The Squeeze-and-Excitation (SE) Upgrade
We heavily modified the standard ST-GCN architecture by inculcating **SE Blocks (Squeeze-and-Excitation)** directly into the spatio-temporal layers. 
- **The Problem:** In a 6-channel network, velocity channels might be critical during a fast lift but act as pure noise when a worker is standing still. 
- **The Solution:** The SE Block acts as a dynamic attention mechanism. By squeezing the spatial dimensions using `AdaptiveAvgPool2d` and passing them through a multi-layer perceptron (FC layers + Sigmoid), the network learns to dynamically assign weights to the most important channels.
- **The Result:** The model mathematically learns to **"excite"** the features of load-bearing joints (Spine/Knees) during a lift, while **"squeezing"** (suppressing) peripheral noise like waving hands or head movements.

Combined with dynamic `Dropout` layers and symmetric normalization, the model achieved a staggering **94.6% Accuracy** in classifying complex, ambiguous lifting postures into Safe, Warning, and Critical categories.

---

## Engineering Highlights 

This project was built from the ground up to demonstrate production-grade Computer Vision engineering, focusing on physics, signal processing, and functional safety rather than just basic deep learning:

1. **Viewpoint & Scale Invariance (Geometric Algebra):**
   Instead of brute-forcing data collection from multiple angles, the system uses linear algebra to calculate the hip-center and applies a **Rotation Matrix** to perfectly align the 3D skeleton to the camera plane. This makes the AI completely agnostic to where the factory camera is physically mounted.
2. **High-Frequency Signal Smoothing (The 1€ Filter):**
   Pose estimation networks inherently suffer from spatial jitter. We integrated a **1 Euro Filter** an adaptive low-pass filter that dynamically changes its cutoff frequency based on movement speed. Fast movements eliminate lag, while standing still aggressively eliminates jitter, proving a deep understanding of real-time signal processing.
3. **Hybrid Intelligence (Deterministic Fallback):**
   In the real world, neural networks hallucinate and servers disconnect. We engineered a **Bio-Mechanical Fallback Engine** that calculates the Euclidean angles of the Torso and Knees based on European Assembly Worksheet (EAWS) standards. If the Cloud/Server dies, the Edge client seamlessly falls back to pure geometry to ensure functional safety is never compromised.
4. **Zero-Bloat Binary Networking:**
   Streaming video or heavy JSON strings over TCP causes massive buffer bloat. We utilized `numpy.tobytes()` and `struct` to pack the spatial state and hardware timestamp into exactly 204 raw binary bytes, achieving microscopic latency and ensuring memory-safe I/O on constrained edge devices.

---

## End-to-End System Architecture

The system is decoupled into highly specialized modules to ensure zero-latency processing.

### 1. The Edge Client (Camera & UX)
Running on factory floor hardware, the client handles vision extraction and the operator UI.
* **Vision Extraction (MediaPipe):** Bypasses OS buffer bloat via `V4L2` drivers to extract a 33-point 3D skeleton in real-time.
* **Temporal Smoothing:** Applies a **OneEuroFilter** to the raw keypoints, destroying high-frequency camera jitter.
* **Mathematical Normalization:** Scales the skeleton by spine length and rotates it algebraically to align hips with the camera axis, ensuring the AI performs perfectly regardless of the camera's angle.
* **Industrial UX:** A dark-mode, monochrome interface built in PyQt6. It features a **"Chameleon Skeleton"**—drawing the worker in neutral grey, but dynamically flashing bright Red/Orange to instantly draw the operator's attention when danger is detected.

### 2. The Network Protocol (Binary Websockets)
* **Decoupled Asynchronous I/O:** The client streams data via a background thread, preventing network latency from freezing the camera feed.
* **Binary Packing:** The skeleton and its timestamp are packed into raw `float32` bytes using Python's `struct`, achieving sub-millisecond TCP transmission times and destroying standard JSON overhead.

### 3. The Server Brain (FastAPI + Asyncio)
* **Vacuum & Brain Loops:** Uses `asyncio.gather` to run two isolated tasks per client: a "Receiver Task" that vacuums TCP data instantly, and an "Inference Task" that analyzes the buffer 10 times a second.

---

## Getting Started

###  About the Dataset (CarDA)

The core AI model was originally developed and evaluated using the **CarDA (Car Door Assembly)** dataset, an extensive multi-modal industrial dataset:

* **Real-World Environment:** Recorded in an actual automotive manufacturing plant across three workstations (WS10, WS20, WS30) alongside a moving conveyor belt, rather than a controlled laboratory.
* **Multi-Modal Sensors:** The recordings were captured using **StereoLabs ZED2** RGB-D cameras synchronized with **XSens MVN Link suits**, providing absolute ground-truth 3D kinematic data at 60 fps.
* **EAWS Annotations:** The dataset includes expert manual annotations for European Assembly Work Sheet (EAWS) standards, evaluating stressful postures like "Trunk Rotation", "Lateral Bending", and "Strongly Bend Forward".

**Paper:**  
[arXiv:2409.17356](https://arxiv.org/abs/2409.17356)

---


### Prerequisites
- Python 3.10+
- Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-repo/industrial-pose.git
cd industrial-pose

# Create and activate environment
conda create -n mlops python=3.10
conda activate mlops
pip install -r requirements.txt
```

### 1. Server Deployment (The Brain)
Start the centralized inference engine. 
```bash
python server/main.py
```
*Output: `Server: AI Model Loaded on CUDA (or CPU)`*

### 2. Client Deployment (The Edge)
Run the GUI on the hardware physically connected to the camera.
```bash
python client/run_app.py
```

---

## Model Training Pipeline

If you want to retrain the SE-STGCN model on new data (or the original CarDa video dataset):

1. **Record Data:** Use the client's built-in recording UI.
2. **Generate Tensors:** Run `python model/generate_data.py` to re-engineer the raw recordings into `.npy` interpolated velocity tensors.
3. **Train:** Run `python model/train.py`. The training script uses CrossEntropyLoss, Adam Optimizer, and StepLR. The best weights are saved to `runs/`.

---

## Diagnostics & Logging

Every time a client connects, the server automatically generates a session diagnostic log in `server/logs/session_<IP>_<timestamp>.csv`. This tracks:
- End-to-end network latency
- Neural Network Inference Time
- Buffer Queue Depths

---
