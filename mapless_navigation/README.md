# RNN-based Mapless Navigation for ROS2 Jazzy

This package provides a ROS2 node that uses a trained torch model for reactive navigation. The model takes a sequence of laser scans and outputs linear/angular velocity commands.

## Prerequisites
- **Python 3.12**
- **Pip** (installed via `sudo apt install python3-pip`)
- **NVIDIA GPU** (optional, for CUDA acceleration) with driver **≥ 530** (for CUDA 12.1 support)

## Installation

### 1. Ensure NVIDIA driver is installed (if using GPU)

Check your current driver version:

```bash
nvidia-smi
```

### 2. Install PyTorch with CUDA 12.1 support


Run the following command to install torch using the provided installation script:
  ```bash
  ./torch_install.sh
  ```