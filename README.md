<!-- markdownlint-disable MD033 -->
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
<summary>Standard Setup with Docker, noVNC, and Foxglove</summary>

This was tested on Windows but should work on Mac. Linux will be different, so make changes as needed.

Feel free to ask for help in the Racerbot Discord Server.

### 0) Install prerequisites (Windows)

#### A) Install Docker Desktop

1) Download: [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2) Install with defaults (Linux containers enabled).
3) Start Docker Desktop and wait until it says "Running".

#### B) Install VS Code

1) Download: [VS Code](https://code.visualstudio.com/)
2) Install with defaults.

#### C) Install Git

1) Download: [Git](https://git-scm.com/downloads)
2) Install with defaults.

### 1) Clone the modified repo

Open PowerShell and run (or create a folder where you want it and `cd` into it):

```powershell
mkdir d:\RacerBot
cd d:\RacerBot

git clone -b main https://github.com/Sighton-GH/SageMath-Webcode.git f1tenth_gym_ros
```

Open the folder in VS Code:

```powershell
code d:\RacerBot\f1tenth_gym_ros
```

### 2) Verify required folders exist

Make sure this folder exists:

```text
d:\RacerBot\f1tenth_gym_ros\f1tenth_gym\f1tenth_gym
```

If it does not exist, stop and ask for help before continuing.

### 3) Build the Docker image

```powershell
cd d:\RacerBot\f1tenth_gym_ros

docker compose build
```

docker compose build --build-arg ENABLE_ARM_QT=0
docker compose build --build-arg ENABLE_ARM_QT=1
Apple Silicon (mac) note:

- The compose file forces linux/amd64 to avoid PyQt6 build failures on arm64.

### 4) Start containers

```powershell
docker compose up -d
```

You should now have:

- f1tenth_gym_ros-sim-1
- f1tenth_gym_ros-novnc-1

### 5) Open the noVNC display

Open in your browser:

```text
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

1) Open [Foxglove](https://app.foxglove.dev).
2) Create an account.
3) Add a connection:
   - Connection type: Foxglove WebSocket
   - Select ROS 2
   - URL: ws://localhost:8765
4) Import the layout file:
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

1) Clone `f1tenth_gym_ros` into the workspace (dev-humble):

```bash
cd $HOME/sim_ws/src
git clone -b dev-humble https://github.com/f1tenth/f1tenth_gym_ros.git
```

1) Clone `f1tenth_gym` inside `f1tenth_gym_ros` (dev-humble):

```bash
cd $HOME/sim_ws/src/f1tenth_gym_ros
git clone -b dev-humble https://github.com/f1tenth/f1tenth_gym.git
```

If the `f1tenth_gym` directory already exists (some clones include it), skip the clone.

1) Install `f1tenth_gym`:

```bash
cd $HOME/sim_ws/src/f1tenth_gym_ros/f1tenth_gym/f1tenth_gym
pip install -e .
```

1) Install ROS dependencies and build the workspace:

```bash
source /opt/ros/humble/setup.bash
cd $HOME/sim_ws
rosdep install -i --from-path src --rosdistro humble -y
colcon build
```

Once you're done, continue to Launching the Simulation below.

</details>

<details>
<summary>Docker with an NVIDIA GPU (better performance)</summary>

**Dependencies:**

- [Docker](https://docs.docker.com/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Rocker](https://github.com/osrf/rocker)
- Windows users must use WSL2 for GPU support.

**Install and run:**

```bash
cd f1tenth_gym_ros

docker build -t f1tenth_gym_ros -f Dockerfile .
rocker --nvidia --x11 --volume .:/sim_ws/src/f1tenth_gym_ros -- f1tenth_gym_ros
```

</details>

<details>
<summary>Original instructions from RoboRacer (OLD)</summary>

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

- [noVNC](http://localhost:8080/vnc.html)

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

## Configuration

Settings that change how the simulation behaves.

- Config file: f1tenth_gym_ros/config/sim.yaml
- Map path: can be maps/levine or a built-in gym track name like Spielberg.
- num_agent: 1 or 2 (multi-agent >2 not supported yet).

### Using a different map

1) Edit `map_path` in f1tenth_gym_ros/config/sim.yaml:
   - Example package-relative: `maps/levine`
   - Example built-in gym track: `Spielberg`

2) Rebuild inside the container:

```powershell
docker exec -it f1tenth_gym_ros-sim-1 /bin/bash
```

```bash
source /opt/ros/humble/setup.bash
cd /sim_ws
colcon build --symlink-install
```

1) Relaunch the bridge:

```bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

Notes:

- For custom maps, keep the .yaml and .png together in the maps folder.
- Example: maps/levine resolves to maps/levine.yaml + maps/levine.png.

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

## Troubleshooting

Common setup issues and fixes.

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

docker compose build --build-arg ENABLE_ARM_QT=1
docker run --rm -it f1tenth_gym_ros qmake -v
docker compose build --no-cache --build-arg ENABLE_ARM_QT=1
Fix: use linux/amd64 emulation (set in docker-compose.yml).

If qmake is missing, the Qt packages did not install. Rebuild without cache:

```bash
docker compose build --no-cache --build-arg ENABLE_ARM_QT=1
```

## Build speed tips

- Docker BuildKit cache is enabled for APT and pip.
- A .dockerignore file reduces build context size.
- Rebuilds will reuse cached layers.
- Rebuilds will reuse cached layers.- A .dockerignore file reduces build context size.- Docker BuildKit cache is enabled for APT and pip.## Build speed tips```docker compose build --no-cache --build-arg ENABLE_ARM_QT=1```bashIf qmake is missing, the Qt packages did not install. Rebuild without cache:```docker run --rm -it f1tenth_gym_ros qmake -v```bashIf it still fails, verify qmake exists inside the image:```docker compose build --build-arg ENABLE_ARM_QT=1```bashFix: rebuild (Qt dev tools auto-install on arm64), or force enable if disabled:### PyQt6 build error on Apple Silicon (arm64)Fix: install f1tenth_gym into system Python inside the image (handled in Dockerfile).### ModuleNotFoundError: No module named f1tenth_gymFix: install uv_build in system Python before installing the gym package in system Python (handled in Dockerfile).### Build error: uv-build not properly installed```/sim_ws/src/f1tenth_gym_ros/f1tenth_gym/f1tenth_gym```textCause: the Gym repo is nested. Use the path:### Build error: editable install path not found- Reconnect to ws://localhost:8765.- Ensure port 8765 is published in docker-compose.yml.- Use connection type Foxglove WebSocket (not ROS/rosbridge).### Foxglove connection failedCommon setup issues and fixes.## Troubleshooting</details>```  -p speed_step:=0.5 -p steer_step:=0.1 -p max_speed:=5.0 -p max_steer:=0.4ros2 run f1tenth_gym_ros wasd_teleop --ros-args \```bashTuning (optional):- Q / E: fine steering trim- R: reset speed + steering- Space: stop- D / Right Arrow: steer right- A / Left Arrow: steer left- S / Down Arrow: decrease speed (reverse)- W / Up Arrow: increase speed (forward)Key mapping:```ros2 run f1tenth_gym_ros wasd_teleop```bashRun inside the sim container:<summary>WASD + Arrow key teleop (recommended)</summary><details></details>```ros2 run teleop_twist_keyboard teleop_twist_keyboard```bash2) Run:1) Set kb_teleop: true in f1tenth_gym_ros/config/sim.yaml<summary>Keyboard teleop (default)</summary><details>- Example: maps/levine resolves to maps/levine.yaml + maps/levine.png.- For custom maps, keep the .yaml and .png together in the maps folder.Notes:```ros2 launch f1tenth_gym_ros gym_bridge_launch.py```bash3) Relaunch the bridge:```colcon build --symlink-installcd /sim_wssource /opt/ros/humble/setup.bash```bash```docker exec -it f1tenth_gym_ros-sim-1 /bin/bash```powershell2) Rebuild inside the container:   - Example built-in gym track: `Spielberg`   - Example package-relative: `maps/levine`1) Edit `map_path` in f1tenth_gym_ros/config/sim.yaml:### Using a different map- num_agent: 1 or 2 (multi-agent >2 not supported yet).- Map path: can be maps/levine or a built-in gym track name like Spielberg.- Config file: f1tenth_gym_ros/config/sim.yamlSettings that change how the simulation behaves.## Configuration</details>```ros2 launch f1tenth_gym_ros gym_bridge_launch.pysource /sim_ws/install/local_setup.bashsource /opt/ros/humble/setup.bashsource $HOME/sim_ws/.venv/bin/activate```bash<summary>Launching the Simulation (any setup)</summary><details></details>- http://localhost:8080/vnc.htmlOpen noVNC:```docker exec -it f1tenth_gym_ros-sim-1 /bin/bash```bashShell into the sim container:```docker compose up```bash**Install and run:**- Docker Compose- Docker**Dependencies:**<summary>Original instructions from RoboRacer (OLD)</summary><details></details>```rocker --nvidia --x11 --volume .:/sim_ws/src/f1tenth_gym_ros -- f1tenth_gym_rosdocker build -t f1tenth_gym_ros -f Dockerfile .cd f1tenth_gym_ros```bash**Install and run:**- Windows users must use WSL2 for GPU support.- [Rocker](https://github.com/osrf/rocker)- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)- [Docker](https://docs.docker.com/install/)**Dependencies:**<summary>Docker with an NVIDIA GPU (better performance)</summary><details></details>Once you're done, continue to Launching the Simulation below.```colcon buildrosdep install -i --from-path src --rosdistro humble -ycd $HOME/sim_wssource /opt/ros/humble/setup.bash```bash6) Install ROS dependencies and build the workspace:```pip install -e .cd $HOME/sim_ws/src/f1tenth_gym_ros/f1tenth_gym/f1tenth_gym```bash5) Install `f1tenth_gym`:If the `f1tenth_gym` directory already exists (some clones include it), skip the clone.```git clone -b dev-humble https://github.com/f1tenth/f1tenth_gym.gitcd $HOME/sim_ws/src/f1tenth_gym_ros```bash4) Clone `f1tenth_gym` inside `f1tenth_gym_ros` (dev-humble):```git clone -b dev-humble https://github.com/f1tenth/f1tenth_gym_ros.gitcd $HOME/sim_ws/src```bash3) Clone `f1tenth_gym_ros` into the workspace (dev-humble):```python3 -m pip install -U pipsource $HOME/sim_ws/.venv/bin/activatepython3 -m venv --system-site-packages $HOME/sim_ws/.venvmkdir -p $HOME/sim_ws/src```bash2) Create the workspace and a venv inside it:1) Install ROS 2 Humble: [ROS 2 Humble Install](https://docs.ros.org/en/humble/Installation.html)### Step-by-step install (fresh workspace)<summary>Native on Ubuntu 22.04 (best performance)</summary><details></details>3) Click at the origin and drag to set heading.2) Use the **2D Pose Estimate** tool.1) In the 3D panel settings, set Fixed frame to `map`.Use Foxglove:### 9) Reset the car to origin- Q / E: fine steering trim- R: reset speed + steering- Space: stop- D / Right Arrow: steer right- A / Left Arrow: steer left- S / Down Arrow: reverse- W / Up Arrow: forwardKey mapping:```ros2 run f1tenth_gym_ros wasd_teleopsource /sim_ws/install/local_setup.bashsource /opt/ros/humble/setup.bash```bashThen:```docker exec -it f1tenth_gym_ros-sim-1 /bin/bash```powershellRun inside the sim container:### 8) Drive the car (WASD + arrows)   - d:\RacerBot\f1tenth_gym_ros\launch\gym_bridge_foxglove.json4) Import the layout file:   - URL: ws://localhost:8765   - Select ROS 2   - Connection type: Foxglove WebSocket3) Add a connection:2) Create an account.1) Open [Foxglove](https://app.foxglove.dev).### 7) Connect Foxglove```docker exec f1tenth_gym_ros-sim-1 bash -lc "tail -n 50 /tmp/gym_bridge.log"```powershellOptional: check logs```docker exec -d f1tenth_gym_ros-sim-1 bash -lc "source /sim_ws/.venv/bin/activate; source /opt/ros/humble/setup.bash; source /sim_ws/install/local_setup.bash; ros2 launch f1tenth_gym_ros gym_bridge_launch.py 2>&1 | tee /tmp/gym_bridge.log"```powershell### 6) Launch the simulationClick **Connect**.```http://localhost:8080/vnc.html```textOpen in your browser:### 5) Open the noVNC display- f1tenth_gym_ros-novnc-1- f1tenth_gym_ros-sim-1You should now have:```docker compose up -d```powershell### 4) Start containers```docker compose build --build-arg ENABLE_ARM_QT=1```bash- To force install Qt deps (if build still fails), use:```docker compose build --build-arg ENABLE_ARM_QT=0```bash- To disable (faster build, but PyQt6 may fail), use:- Qt dev tools are installed automatically on arm64 builds.Apple Silicon (mac) note:```docker compose buildcd d:\RacerBot\f1tenth_gym_ros```powershell### 3) Build the Docker imageIf it does not exist, stop and ask for help before continuing.```d:\RacerBot\f1tenth_gym_ros\f1tenth_gym\f1tenth_gym```textMake sure this folder exists:### 2) Verify required folders exist```code d:\RacerBot\f1tenth_gym_ros```powershellOpen the folder in VS Code:```git clone -b main https://github.com/Sighton-GH/SageMath-Webcode.git f1tenth_gym_roscd d:\RacerBotmkdir d:\RacerBot```powershellOpen PowerShell and run (or create a folder where you want it and `cd` into it):### 1) Clone the modified repo2) Install with defaults.1) Download: [Git](https://git-scm.com/downloads)#### C) Install Git2) Install with defaults.1) Download: [VS Code](https://code.visualstudio.com/)#### B) Install VS Code3) Start Docker Desktop and wait until it says "Running".2) Install with defaults (Linux containers enabled).1) Download: [Docker Desktop](https://www.docker.com/products/docker-desktop/)#### A) Install Docker Desktop### 0) Install prerequisites (Windows)Feel free to ask for help in the Racerbot Discord Server.This was tested on Windows but should work on Mac. Linux will be different, so make changes as needed.<summary>Standard Setup with Docker, noVNC, and Foxglove</summary><details>- Recommended for performance: Native Ubuntu 22.04 with ROS 2 or Docker with NVIDIA GPU.- Recommended for simplicity: Windows + Docker Desktop + noVNC.Pick the option that matches your platform and goals.## Setup options- Windows 10/11, macOS, and Ubuntu without an NVIDIA GPU (using noVNC).- Windows 10/11, macOS, and Ubuntu with an NVIDIA GPU (via WSL2 and NVIDIA Container Toolkit).- Ubuntu native (tested on 22.04 and 24.04) with ROS 2.These are the supported OS and runtime combinations.## Supported systemsIf you are new to ROS or Docker, start with the Windows + Docker Desktop + noVNC section.This repository provides a ROS 2 bridge that turns the F1TENTH Gym environment into a ROS2 simulation.# F1TENTH Gym ROS2 communication bridgeThis repository provides a ROS 2 bridge that turns the F1TENTH Gym environment into a ROS2 simulation.

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
<summary>Standard Setup with Docker, noVNC, and Foxglove</summary>

This was tested on windows but should work on Mac. Linux will be different so make changes as needed.

Feel free to ask for help in the Racerbot Discord Server.

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

### 1) Clone the modified repo
Open PowerShell and run:
Or create a folder where you want it and cd into it in powershell 
```powershell
mkdir d:\RacerBot
cd d:\RacerBot

git clone -b main https://github.com/Sighton-GH/SageMath-Webcode.git f1tenth_gym_ros
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

Apple Silicon (mac) note:
- Qt dev tools are installed automatically on arm64 builds.
- To disable (faster build, but PyQt6 may fail), use:
   ```bash
   docker compose build --build-arg ENABLE_ARM_QT=0
   ```
- To force install Qt deps (if build still fails), use:
   ```bash
   docker compose build --build-arg ENABLE_ARM_QT=1
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
2) Create an account
3) Add a connection:
   - Connection type: Foxglove WebSocket
   - Select ros2
   - URL: ws://localhost:8765
4) Import the layout file:
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
<summary>Native on Ubuntu 22.04 (best performance)</summary>

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
<summary>Docker with an NVIDIA GPU (better performance)</summary>

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
<summary>Original instructions from RoboRacer (OLD)</summary>

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

## Configuration
Settings that change how the simulation behaves.
- Config file: f1tenth_gym_ros/config/sim.yaml
- Map path: can be maps/levine or a built-in gym track name like Spielberg.
- num_agent: 1 or 2 (multi-agent >2 not supported yet).

### Using a different map
1) Edit `map_path` in f1tenth_gym_ros/config/sim.yaml:
- Example package-relative: `maps/levine`
- Example built-in gym track: `Spielberg`

2) Rebuild inside the container:
```powershell
docker exec -it f1tenth_gym_ros-sim-1 /bin/bash
```
```bash
source /opt/ros/humble/setup.bash
cd /sim_ws
colcon build --symlink-install
```

3) Relaunch the bridge:
```bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

Notes:
- For custom maps, keep the .yaml and .png together in the maps folder.
- Example: maps/levine resolves to maps/levine.yaml + maps/levine.png.

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

## Troubleshooting
Common setup issues and fixes.

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

### PyQt6 build error on Apple Silicon (arm64)
Fix: rebuild (Qt dev tools auto-install on arm64), or force enable if disabled:
```bash
docker compose build --build-arg ENABLE_ARM_QT=1
```

If it still fails, verify qmake exists inside the image:
```bash
docker run --rm -it f1tenth_gym_ros qmake -v
```
If qmake is missing, the Qt packages did not install. Rebuild without cache:
```bash
docker compose build --no-cache --build-arg ENABLE_ARM_QT=1
```

## Build speed tips
- Docker BuildKit cache is enabled for APT and pip.
- A .dockerignore file reduces build context size.
- Rebuilds will reuse cached layers.
