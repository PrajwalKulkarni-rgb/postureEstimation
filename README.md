# Industrial Posture Analysis & Safety Monitor

## Overview

In industrial environments, manual material handling is a leading cause of musculoskeletal disorders. This project is a real-time safety monitor designed to catch dangerous lifting postures before they cause injury. 

Instead of relying on basic angles or static images, we built an end-to-end system that tracks the worker's 3D skeleton, calculates their movement physics, and feeds it into an advanced Deep Learning model (ST-GCN) across an edge-to-cloud network.

---

## 1. How We Process the Data

Standard AI models often break when a camera stutters or the frame rate changes. We had to build a custom data pipeline to handle real-world factory conditions:

- **Hardware Timestamping:** Every skeletal frame is stamped with a precise physical timestamp the microsecond it leaves the camera.
- **Temporal Interpolation:** When the server receives a 5-second chunk of video, it doesn't matter if it has 50 frames or 150 frames. We mathematically resample the timeline into exactly **50 uniform frames**. This makes the model completely immune to camera stutter and frame drops.
- **Adding Physics (Velocity):** Static positions don't tell the whole story. A fast lift is much more dangerous than a slow lift. We calculate the true velocity vectors (`V = dS / dt`) for every joint. 
- **The Input:** The final input to the AI isn't an image. It's a dense 6-channel tensor (`X, Y, Z, Vx, Vy, Vz`) representing the complete physical state of the worker.

---

## 2. The AI Architecture (SE-STGCN)

The brain of the system is a **Spatial-Temporal Graph Convolutional Network (ST-GCN)**. It treats the human body as a connected graph (bones and joints) rather than a flat image.

**The Squeeze-and-Excitation Upgrade:**
We modified the standard ST-GCN by adding **SE Blocks (Attention)**. 
Because we pass 6 channels (positions + velocities) into the network, velocity might be critical during a fast lift but act as pure noise when a worker is standing still. The SE blocks act as an attention mechanism, allowing the network to dynamically "excite" important channels and "squeeze" out irrelevant noise depending on what the worker is doing.

**Current Results:**
The model currently achieves **93.65% Validation Accuracy** in classifying complex lifting postures (Safe, Warning, Critical). This was achieved while training against aggressive physical augmentations—including simulated camera rotations, random joint jitter, and simulated occlusions (missing limbs)—making the model highly robust for real-world deployment.

---

## 3. Engineering Highlights

This project was built to be a production-grade system, focusing heavily on signal processing and functional safety:

- **Angle/Viewpoint Invariance:** We use linear algebra to automatically rotate the skeleton so the worker's hips are always aligned with the camera axis. This means you can mount the camera anywhere (front, side, diagonal) and the AI will still understand the posture perfectly.
- **The 1 Euro Filter:** Raw camera skeletons vibrate and jitter, which ruins velocity calculations. We integrated a 1 Euro Filter, which dynamically changes how much it smooths the data based on movement speed. It kills jitter when standing still, but prevents lag when moving fast.
- **Bio-Mechanical Fallback:** If the AI server crashes or the network drops, the Edge client seamlessly falls back to a deterministic geometry engine that calculates torso angles based on EAWS standards. Functional safety is never compromised.
- **Binary WebSockets:** Sending JSON over the network is too slow for real-time edge devices. We pack the skeleton data into raw binary `float32` bytes, crushing the data size down to 204 bytes per frame and achieving sub-millisecond latency.

---

## 4. End-to-End System

The system is decoupled into two parts:

### The Edge Client (Camera & UI)
Runs on the factory floor. It extracts the skeleton using MediaPipe, applies the 1 Euro filter, normalizes the coordinates, and streams the binary data to the server. It features a dark-mode UI that flashes warning colors when danger is detected.

### The Server Brain (FastAPI)
Runs the PyTorch ST-GCN model. It uses Python `asyncio` to vacuum up the binary TCP data instantly while analyzing the buffer in a separate thread. 

---

## 5. Getting Started

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

### Server Deployment (The Brain)
Start the centralized inference engine. 
```bash
python server/main.py
```
*Output: `Server: AI Model Loaded on CUDA (or CPU)`*

### Client Deployment (The Edge)
Run the GUI on the hardware physically connected to the camera.
```bash
python client/run_app.py
```

### Model Training
To retrain the model on new data:
1. Run `python model/generate_data.py` to extract tensors from your `.mp4` files.
2. Run `python model/train.py` to start training. The best weights are automatically saved to `runs/`.

---

## About the Dataset (CarDA)
The core AI model was originally developed using the **CarDA (Car Door Assembly)** dataset. It features real-world factory recordings alongside a moving conveyor belt, captured with StereoLabs ZED2 cameras and XSens MVN Link suits for absolute ground-truth 3D kinematics at 60 fps. It includes expert manual annotations for European Assembly Work Sheet (EAWS) standards.

**Paper:** [arXiv:2409.17356](https://arxiv.org/abs/2409.17356)
