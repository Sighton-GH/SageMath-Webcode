<!-- markdownlint-disable MD033 -->
# F1TENTH Gym ROS2 communication bridge

This repository provides a ROS 2 bridge that turns the F1TENTH Gym environment into a ROS 2 simulation.

If you are new to ROS or Docker, start with the Windows + Docker Desktop + noVNC steps in Setup options.

## Table of contents

- [Supported systems](#supported-systems)
- [Setup options](#setup-options)
- [Launching the simulation](#launching-the-simulation)
- [Ports and URLs](#ports-and-urls)
- [Simulation settings](#simulation-settings)
- [Maps](#maps)
- [Teleop](#teleop)
- [Topics](#topics)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Developing your own agent](#developing-your-own-agent)

## Supported systems

- Ubuntu native (tested on 22.04 and 24.04) with ROS 2.
- Windows 10/11, macOS, and Ubuntu with an NVIDIA GPU (via WSL2 and NVIDIA Container Toolkit).
- Windows 10/11, macOS, and Ubuntu without an NVIDIA GPU (using noVNC).

## Setup options

<details>
<summary>Standard setup with Docker, noVNC, and Foxglove (recommended)</summary>

This was tested on Windows but should work on Mac. Linux will be different, so make changes as needed.

### Quick start (Windows + Docker + noVNC)

1) Install prerequisites: Docker Desktop, VS Code, Git.
2) Clone and open the repo.
3) Build and start containers.
4) Open noVNC and launch the bridge.
5) Connect Foxglove and import the config launch JSON file.
6) Drive with WASD.

### 1) Install prerequisites

- Docker Desktop: [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- VS Code: [VS Code](https://code.visualstudio.com/)
- Git: [Git](https://git-scm.com/downloads)

### 2) Clone the repo

```powershell
mkdir d:\RacerBot
cd d:\RacerBot

git clone -b main https://github.com/Sighton-GH/SageMath-Webcode.git f1tenth_gym_ros
code d:\RacerBot\f1tenth_gym_ros
```

### 3) Verify required folders exist

Make sure this folder exists:

```text
d:\RacerBot\f1tenth_gym_ros\f1tenth_gym\f1tenth_gym
```

If it does not exist, stop and ask for help before continuing.

### 4) Build and start containers

```powershell
cd d:\RacerBot\f1tenth_gym_ros

docker compose build
docker compose up -d
```

You should now have:

- f1tenth_gym_ros-sim-1
- f1tenth_gym_ros-novnc-1

### 5) Open noVNC

Open in your browser:

```text
http://localhost:8080/vnc.html
```

Click Connect.

### 6) Launch the simulation

```powershell
docker exec -d f1tenth_gym_ros-sim-1 bash -lc "source /sim_ws/.venv/bin/activate; source /opt/ros/humble/setup.bash; source /sim_ws/install/local_setup.bash; ros2 launch f1tenth_gym_ros gym_bridge_launch.py 2>&1 | tee /tmp/gym_bridge.log"
```

Optional log check:

```powershell
docker exec f1tenth_gym_ros-sim-1 bash -lc "tail -n 50 /tmp/gym_bridge.log"
```

### 7) Connect Foxglove

1) Open [Foxglove](https://app.foxglove.dev).
2) Create an account.
3) Add a connection:
  - Connection type: Foxglove WebSocket
  - Select ROS 2
  - URL: ws://localhost:8765
4) Import the config launch JSON file:
  - d:\RacerBot\f1tenth_gym_ros\launch\gym_bridge_foxglove.json

### 8) Drive the car (WASD + arrows)

Run inside the sim container:

```powershell
docker exec -it f1tenth_gym_ros-sim-1 /bin/bash
```

Then:

```bash
source /opt/ros/humble/setup.bash
source /sim_ws/install/local_setup.bash
ros2 run f1tenth_gym_ros wasd_teleop
```

Key mapping:

- W / Up Arrow: forward
- S / Down Arrow: reverse
- A / Left Arrow: steer left
- D / Right Arrow: steer right
- Space: stop
- R: reset speed + steering
- Q / E: fine steering trim

### 9) Reset the car to origin

Use Foxglove:

1) In the 3D panel settings, set Fixed frame to map.
2) Use the 2D Pose Estimate tool.
3) Click at the origin and drag to set heading.

Apple Silicon (mac) note:

- The compose file forces linux/amd64 to avoid PyQt6 build failures on arm64.
- To disable Qt dev tools (faster build, but PyQt6 may fail):
  ```bash
  docker compose build --build-arg ENABLE_ARM_QT=0
  ```
- To force install Qt dev tools:
  ```bash
  docker compose build --build-arg ENABLE_ARM_QT=1
  ```

</details>

<details>
<summary>Native on Ubuntu 22.04 (best performance)</summary>

### Step-by-step install (fresh workspace)

1) Install ROS 2 Humble: [ROS 2 Humble Install](https://docs.ros.org/en/humble/Installation.html)
2) Create the workspace and a venv inside it:

```bash
mkdir -p $HOME/sim_ws/src
python3 -m venv --system-site-packages $HOME/sim_ws/.venv
source $HOME/sim_ws/.venv/bin/activate
python3 -m pip install -U pip
```

3) Clone f1tenth_gym_ros into the workspace (dev-humble):

```bash
cd $HOME/sim_ws/src
git clone -b dev-humble https://github.com/f1tenth/f1tenth_gym_ros.git
```

4) Clone f1tenth_gym inside f1tenth_gym_ros (dev-humble):

```bash
cd $HOME/sim_ws/src/f1tenth_gym_ros
git clone -b dev-humble https://github.com/f1tenth/f1tenth_gym.git
```

If the f1tenth_gym directory already exists (some clones include it), skip the clone.

5) Install f1tenth_gym:

```bash
cd $HOME/sim_ws/src/f1tenth_gym_ros/f1tenth_gym/f1tenth_gym
pip install -e .
```

6) Install ROS dependencies and build the workspace:

```bash
source /opt/ros/humble/setup.bash
cd $HOME/sim_ws
rosdep install -i --from-path src --rosdistro humble -y
colcon build
```

Once done, continue to Launching the simulation.

</details>

<details>
<summary>Docker with an NVIDIA GPU (better performance)</summary>

Dependencies:

- [Docker](https://docs.docker.com/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Rocker](https://github.com/osrf/rocker)
- Windows users must use WSL2 for GPU support.

Install and run:

```bash
cd f1tenth_gym_ros

docker build -t f1tenth_gym_ros -f Dockerfile .
rocker --nvidia --x11 --volume .:/sim_ws/src/f1tenth_gym_ros -- f1tenth_gym_ros
```

</details>

<details>
<summary>Original instructions from RoboRacer (OLD)</summary>

Dependencies:

- Docker
- Docker Compose

Install and run:

```bash
docker compose up
```

Shell into the sim container:

```bash
docker exec -it f1tenth_gym_ros-sim-1 /bin/bash
```

Open noVNC:

- [noVNC](http://localhost:8080/vnc.html)

</details>

## Launching the simulation

```bash
source $HOME/sim_ws/.venv/bin/activate
source /opt/ros/humble/setup.bash
source /sim_ws/install/local_setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

After the bridge is running, connect Foxglove and import the config launch JSON file. It preloads the recommended panels, topics, and layout for the sim:

- d:\RacerBot\f1tenth_gym_ros\launch\gym_bridge_foxglove.json

## Ports and URLs

- noVNC: http://localhost:8080/vnc.html
- Foxglove WebSocket: ws://localhost:8765
- Foxglove app: https://app.foxglove.dev
- Layout file: d:\RacerBot\f1tenth_gym_ros\launch\gym_bridge_foxglove.json

## Simulation settings

Settings live in f1tenth_gym_ros/config/sim.yaml.

- Map path: maps/levine or a built-in gym track name like Spielberg.
- num_agent: 1 or 2 (multi-agent >2 not supported yet).
- kb_teleop: true or false.

After changing settings, rebuild inside the container:

```powershell
docker exec -it f1tenth_gym_ros-sim-1 /bin/bash
```

```bash
source /opt/ros/humble/setup.bash
cd /sim_ws
colcon build --symlink-install
```

Relaunch the bridge:

```bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

## Maps

To use a different map:

1) Edit map_path in f1tenth_gym_ros/config/sim.yaml:
   - Example package-relative: maps/levine
   - Example built-in gym track: Spielberg
2) Rebuild and relaunch the bridge (see Simulation settings above).

Notes:

- For custom maps, keep the .yaml and .png together in the maps folder.
- Example: maps/levine resolves to maps/levine.yaml + maps/levine.png.

## Teleop

<details>
<summary>Keyboard teleop (teleop_twist_keyboard)</summary>

1) Set kb_teleop: true in f1tenth_gym_ros/config/sim.yaml
2) Run:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

</details>

<details>
<summary>WASD + Arrow key teleop (recommended)</summary>

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

</details>

## Topics

Published (single agent):

- /scan: ego laser scan
- /ego_racecar/odom: ego odometry
- /map: environment map
- tf: transform tree

Published (two agents) adds:

- /opp_scan: opponent laser scan
- /ego_racecar/opp_odom: opponent odom for ego planner
- /opp_racecar/odom: opponent odometry
- /opp_racecar/opp_odom: ego odom for opponent planner

Subscribed (single agent):

- /drive: ego drive command (AckermannDriveStamped)
- /initialpose: reset ego pose via 2D Pose Estimate tool

Subscribed (two agents) adds:

- /opp_drive: opponent drive command (AckermannDriveStamped)
- /goal_pose: reset opponent pose via 2D Goal Pose tool

## Troubleshooting

### Foxglove connection failed

- Use connection type Foxglove WebSocket (not ROS/rosbridge).
- Ensure port 8765 is published in docker-compose.yml.
- Reconnect to ws://localhost:8765.

### Build error: editable install path not found

Cause: the Gym repo is nested. Use the path:

```text
/sim_ws/src/f1tenth_gym_ros/f1tenth_gym/f1tenth_gym
```

### Build error: uv-build not properly installed

Fix: install uv_build in system Python before installing the gym package in system Python (handled in Dockerfile).

### ModuleNotFoundError: No module named f1tenth_gym

Fix: install f1tenth_gym into system Python inside the image (handled in Dockerfile).

### PyQt6 build error on Apple Silicon (arm64)

Fix: use linux/amd64 emulation (set in docker-compose.yml).

If qmake is missing, the Qt packages did not install. Rebuild without cache:

```bash
docker compose build --no-cache --build-arg ENABLE_ARM_QT=1
```

## FAQ

### I have Python < 3.9

The current f1tenth_gym requires Python 3.9+. Use Ubuntu 22.04 with ROS 2 Humble or update your Python environment to 3.9+.

### This package is managed externally, PEP 668

You are trying to install the package using the system Python. Use a virtual environment as instructed above.

### PyQt6 6.10 cached, fails to install

Install PyQt6 6.7.1 and then install f1tenth_gym:

```bash
pip3 install pyqt6==6.7.1
pip3 install -e .
```

### Gym install hangs on PyQt6>6.7.1 installation

Install PyQt6 6.7.1 with license options, then install f1tenth_gym:

```bash
pip3 install pyqt6==6.7.1 --config-settings --config-license= --verbose
pip3 install -e .
```

### AttributeError: module 'coverage' has no attribute 'types'

Update coverage:

```bash
pip3 install --upgrade coverage
```

### ImportError: cannot import name 'Transpose' from 'PIL.Image'

Update pillow:

```bash
pip3 install --upgrade pillow
```

### ValueError: numpy.dtype size changed, may indicate binary incompatibility

Update scipy:

```bash
pip3 install --upgrade scipy
```

### "opencv>=3. invalid" error on pip install

Update pip, wheel, and setuptools:

```bash
python3 -m pip install --upgrade pip wheel setuptools
```

## Developing your own agent

You can run your own ROS 2 nodes alongside the sim in two ways:

- Add a new package inside /sim_ws in the sim container and run it in another shell while the sim is running.
- Create a second container for your agent node and attach it to the same Docker network as the sim and noVNC services.

## Build speed tips

- Docker BuildKit cache is enabled for APT and pip.
- A .dockerignore file reduces build context size.
- Rebuilds will reuse cached layers.
