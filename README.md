# F1TENTH Gym ROS2 communication bridge
This repository provides a ROS 2 bridge that turns the F1TENTH Gym environment into a ROS2 simulation.

If you are new to ROS or Docker, start with the Windows + Docker Desktop + noVNC section.

## Supported systems
These are the supported OS and runtime combinations.
- Ubuntu native (tested on 22.04 and 24.04) with ROS 2.
- Windows 10/11, macOS, and Ubuntu with an NVIDIA GPU (via WSL2 and NVIDIA Container Toolkit).
- Windows 10/11, macOS, and Ubuntu without an NVIDIA GPU (using noVNC).

## Setup options
Pick the option that matches your platform and goals.
- Recommended for simplicity: Windows + Docker Desktop + noVNC.
- Recommended for performance: Native Ubuntu 22.04 with ROS 2 or Docker with NVIDIA GPU.

<details>
<summary>Windows + Docker Desktop + noVNC (Beginner friendly, recommended for simplicity)</summary>

### 0) Install prerequisites (Windows)
#### A) Install Docker Desktop
1) Download: https://www.docker.com/products/docker-desktop/
2) Install with defaults (Linux containers enabled).
3) Start Docker Desktop and wait until it says "Running".

#### B) Install VS Code
1) Download: https://code.visualstudio.com/
2) Install with defaults.

#### C) Install Git
1) Download: https://git-scm.com/downloads
2) Install with defaults.

### 1) Clone your repo
Open PowerShell and run:
```powershell
mkdir d:\RacerBot
cd d:\RacerBot

git clone -b dev-humble https://github.com/Sighton-GH/SageMath-Webcode.git f1tenth_gym_ros
```

Open the folder in VS Code:
```powershell
code d:\RacerBot\f1tenth_gym_ros
```

### 2) Verify required folders exist
Make sure this folder exists:
```
d:\RacerBot\f1tenth_gym_ros\f1tenth_gym\f1tenth_gym
```
If it does not exist, stop and ask for help before continuing.

### 3) Build the Docker image
```powershell
cd d:\RacerBot\f1tenth_gym_ros

docker compose build
```

### 4) Start containers
```powershell
docker compose up -d
```

You should now have:
- f1tenth_gym_ros-sim-1
- f1tenth_gym_ros-novnc-1

### 5) Open the noVNC display
Open in your browser:
```
http://localhost:8080/vnc.html
```
Click **Connect**.

### 6) Launch the simulation
```powershell
docker exec -d f1tenth_gym_ros-sim-1 bash -lc "source /sim_ws/.venv/bin/activate; source /opt/ros/humble/setup.bash; source /sim_ws/install/local_setup.bash; ros2 launch f1tenth_gym_ros gym_bridge_launch.py 2>&1 | tee /tmp/gym_bridge.log"
```

Optional: check logs
```powershell
docker exec f1tenth_gym_ros-sim-1 bash -lc "tail -n 50 /tmp/gym_bridge.log"
```

### 7) Connect Foxglove
1) Open https://app.foxglove.dev
2) Add a connection:
   - Connection type: Foxglove WebSocket
   - URL: ws://localhost:8765
3) Import the layout file:
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
1) In the 3D panel settings, set Fixed frame to `map`.
2) Use the **2D Pose Estimate** tool.
3) Click at the origin and drag to set heading.

</details>

<details>
<summary>Native on Ubuntu 22.04 (Recommended for performance)</summary>

### Step-by-step install (fresh workspace)
1. **Install ROS 2 Humble.** Follow the instructions here: https://docs.ros.org/en/humble/Installation.html
2. **Create the workspace and a venv inside it:**
   ```bash
   mkdir -p $HOME/sim_ws/src
   python3 -m venv --system-site-packages $HOME/sim_ws/.venv
   source $HOME/sim_ws/.venv/bin/activate
   python3 -m pip install -U pip
   ```
3. **Clone `f1tenth_gym_ros` into the workspace (dev-humble):**
   ```bash
   cd $HOME/sim_ws/src
   git clone -b dev-humble https://github.com/f1tenth/f1tenth_gym_ros.git
   ```
4. **Clone `f1tenth_gym` inside `f1tenth_gym_ros` (dev-humble):**
   ```bash
   cd $HOME/sim_ws/src/f1tenth_gym_ros
   git clone -b dev-humble https://github.com/f1tenth/f1tenth_gym.git
   ```
   If the `f1tenth_gym` directory already exists (some clones include it), skip the clone.
5. **Install `f1tenth_gym` (follow its README):**
   ```bash
   cd $HOME/sim_ws/src/f1tenth_gym_ros/f1tenth_gym/f1tenth_gym
   pip install -e .
   ```
6. **Install ROS dependencies and build the workspace:**
   ```bash
   source /opt/ros/humble/setup.bash
   cd $HOME/sim_ws
   rosdep install -i --from-path src --rosdistro humble -y
   colcon build
   ```

Once you're done, continue to Launching the Simulation below.

</details>

<details>
<summary>Docker with an NVIDIA GPU (Recommended for performance)</summary>

**Dependencies:**
- Docker: https://docs.docker.com/install/
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
- Rocker: https://github.com/osrf/rocker
- Windows users must use WSL2 for GPU support.

**Install and run:**
```bash
cd f1tenth_gym_ros

docker build -t f1tenth_gym_ros -f Dockerfile .
rocker --nvidia --x11 --volume .:/sim_ws/src/f1tenth_gym_ros -- f1tenth_gym_ros
```

</details>

<details>
<summary>Docker without an NVIDIA GPU (noVNC, simplest cross-platform)</summary>

**Dependencies:**
- Docker
- Docker Compose

**Install and run:**
```bash
docker compose up
```

Shell into the sim container:
```bash
docker exec -it f1tenth_gym_ros-sim-1 /bin/bash
```

Open noVNC:
- http://localhost:8080/vnc.html

</details>

<details>
<summary>Launching the Simulation (any setup)</summary>

```bash
source $HOME/sim_ws/.venv/bin/activate
source /opt/ros/humble/setup.bash
source /sim_ws/install/local_setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

</details>

<details>
<summary>Configuration</summary>

- Config file: f1tenth_gym_ros/config/sim.yaml
- Map path: can be maps/levine or a built-in gym track name like Spielberg.
- num_agent: 1 or 2 (multi-agent >2 not supported yet).

</details>

<details>
<summary>Keyboard teleop (default)</summary>

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

<details>
<summary>Troubleshooting</summary>

### Foxglove connection failed
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

</details>

## Build speed tips
- Docker BuildKit cache is enabled for APT and pip.
- A .dockerignore file reduces build context size.
- Rebuilds will reuse cached layers.
