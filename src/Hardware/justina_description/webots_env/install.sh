#!/bin/bash

# virtual_envs/install.sh - Install simulation dependencies

set -e  # Exit on error

echo "======================================"
echo "Installing Justina Simulation Dependencies"
echo "======================================"

# Update package lists
echo "Updating package lists..."
sudo apt update

# Install ROS 2 Jazzy packages
echo "Installing ROS 2 Jazzy packages..."
sudo apt install -y \
    ros-jazzy-robot-state-publisher \
    ros-jazzy-joint-state-publisher \
    ros-jazzy-joint-state-publisher-gui \
    ros-jazzy-xacro

# Optional but useful packages
echo "Installing optional useful packages..."
sudo apt install -y \
    ros-jazzy-launch-ros \
    ros-jazzy-rviz2 \
    ros-jazzy-tf2-ros \
    ros-jazzy-tf2-tools \
    ros-jazzy-topic-tools \
    ros-jazzy-webots-ros2

# Install Python packages for teleoperation
echo "Installing Python packages for teleoperation..."
sudo apt install -y python3-pygame
sudo apt install ros-jazzy-joy
sudo apt install ros-jazzy-teleop-twist-joy

# ===================== Behavior Tree (PyTrees) =====================
echo "Installing Behavior Tree (PyTrees) packages for ROS 2 Jazzy..."

# Ensure ROS 2 repository is added (in case it's missing)
sudo apt install -y software-properties-common
sudo add-apt-repository universe -y
sudo apt update && sudo apt install -y curl

# Add ROS 2 apt repository (if not already present)
if [ ! -f /usr/share/keyrings/ros-archive-keyring.gpg ]; then
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
    sudo apt update
fi

# Install PyTrees and viewer
sudo apt install -y \
    ros-jazzy-py-trees \
    ros-jazzy-py-trees-ros-interfaces \
    ros-jazzy-py-trees-ros \
    ros-jazzy-py-trees-ros-viewer\
    ros-jazzy-nav2-bringup

# ===================== Clean up =====================
echo "Cleaning up..."
sudo apt autoremove -y

# Verify installations
echo ""
echo "======================================"
echo "Verification"
echo "======================================"

# Check key commands
echo "Checking key commands:"
commands=("ros2" "xacro" "py-trees-tree-viewer")
for cmd in "${commands[@]}"; do
    if command -v $cmd &> /dev/null; then
        echo "✓ $cmd"
    else
        echo "✗ $cmd (may be in /opt/ros/jazzy/bin/)"
    fi
done

# Also check Python import
python3 -c "import py_trees; print('✓ py_trees module')" 2>/dev/null || echo "✗ py_trees module not found"

echo ""
echo "======================================"
echo "Installation complete!"
echo "======================================"