# F1TENTH Gym ROS2 communication bridge
This repository provides a ROS 2 bridge that turns the F1TENTH Gym environment into a ROS2 simulation.

If you are a beginner, start with the Docker Desktop + noVNC path below. It is the most reliable on Windows.

## Supported systems
- Ubuntu native (tested on 22.04 and 24.04) with ROS 2.
- Windows 10/11, macOS, and Ubuntu with an NVIDIA GPU (via WSL2 and NVIDIA Container Toolkit).
- Windows 10/11, macOS, and Ubuntu without an NVIDIA GPU (using noVNC).

## Quick start (Windows + Docker Desktop + noVNC)
This is the easiest path and the one used in the setup notes.

### 1) Clone the repos (dev-humble)
```powershell
mkdir d:\RacerBot
cd d:\RacerBot

git clone -b dev-humble https://github.com/f1tenth/f1tenth_gym_ros.git

cd d:\RacerBot\f1tenth_gym_ros
git clone -b dev-humble https://github.com/f1tenth/f1tenth_gym.git
```

The Gym repo must be nested at:
```
d:\RacerBot\f1tenth_gym_ros\f1tenth_gym\f1tenth_gym
```

### 2) Build the image
```powershell
cd d:\RacerBot\f1tenth_gym_ros
docker compose build
```

### 3) Start containers
```powershell
docker compose up -d
```

### 4) Open noVNC
Open http://localhost:8080/vnc.html and click Connect.

### 5) Launch the simulation
```powershell
docker exec -d f1tenth_gym_ros-sim-1 bash -lc "source /sim_ws/.venv/bin/activate; source /opt/ros/humble/setup.bash; source /sim_ws/install/local_setup.bash; ros2 launch f1tenth_gym_ros gym_bridge_launch.py 2>&1 | tee /tmp/gym_bridge.log"
```

### 6) Connect Foxglove
1) Open https://app.foxglove.dev
2) Add a connection:
   - Connection type: Foxglove WebSocket
   - URL: ws://localhost:8765
3) Import the layout file:
   - d:\RacerBot\f1tenth_gym_ros\launch\gym_bridge_foxglove.json

## Native Ubuntu 22.04 (ROS 2 Humble)
1) Install ROS 2 Humble: https://docs.ros.org/en/humble/Installation.html
2) Create workspace and venv:
```bash
mkdir -p $HOME/sim_ws/src
python3 -m venv --system-site-packages $HOME/sim_ws/.venv
source $HOME/sim_ws/.venv/bin/activate
python3 -m pip install -U pip
```
3) Clone repos:
```bash
cd $HOME/sim_ws/src
git clone -b dev-humble https://github.com/f1tenth/f1tenth_gym_ros.git
cd $HOME/sim_ws/src/f1tenth_gym_ros
git clone -b dev-humble https://github.com/f1tenth/f1tenth_gym.git
```
4) Install Gym:
```bash
cd $HOME/sim_ws/src/f1tenth_gym_ros/f1tenth_gym/f1tenth_gym
pip install -e .
```
5) Install deps and build:
```bash
source /opt/ros/humble/setup.bash
cd $HOME/sim_ws
rosdep install -i --from-path src --rosdistro humble -y
colcon build
```

## Launching the simulation (any setup)
```bash
source $HOME/sim_ws/.venv/bin/activate
source /opt/ros/humble/setup.bash
source /sim_ws/install/local_setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

## Configuration
- Config file: f1tenth_gym_ros/config/sim.yaml
- Map path: can be maps/levine or a built-in gym track name like Spielberg.
- num_agent: 1 or 2 (multi-agent >2 not supported yet).

## Keyboard teleop
1) Set kb_teleop: true in f1tenth_gym_ros/config/sim.yaml
2) Run:
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

## WASD + Arrow key teleop (recommended)
This repo includes a simple keyboard teleop node that uses WASD and arrow keys.

Run inside the sim container:
```bash
ros2 run f1tenth_gym_ros wasd_teleop
```

Key mapping:
- W / Up Arrow: increase speed (forward)
- S / Down Arrow: decrease speed (reverse)
- A / Left Arrow: steer left
- D / Right Arrow: steer right
- Space: stop
- R: reset speed + steering
- Q / E: fine steering trim

Tuning (optional):
```bash
ros2 run f1tenth_gym_ros wasd_teleop --ros-args \
   -p speed_step:=0.5 -p steer_step:=0.1 -p max_speed:=5.0 -p max_steer:=0.4
```

## Troubleshooting
### Foxglove connection failed
Fixes:
- Use connection type Foxglove WebSocket (not ROS/rosbridge).
- Ensure port 8765 is published in docker-compose.yml.
- Reconnect to ws://localhost:8765.

### Build error: editable install path not found
Cause: the Gym repo is nested. Use the path:
```
/sim_ws/src/f1tenth_gym_ros/f1tenth_gym/f1tenth_gym
```

### Build error: uv-build not properly installed
Fix: install uv_build in system Python before installing the gym package in system Python (handled in Dockerfile).

### ModuleNotFoundError: No module named f1tenth_gym
Fix: install f1tenth_gym into system Python inside the image (handled in Dockerfile).

## Build speed tips
- Docker BuildKit cache is enabled for APT and pip.
- A .dockerignore file reduces build context size.
- Rebuilds will reuse cached layers.
