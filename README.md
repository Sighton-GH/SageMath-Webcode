# F1TENTH Gym ROS2 Bridge (Beginner Setup Guide)
This repository provides a ROS 2 bridge that turns the F1TENTH Gym environment into a ROS2 simulation.

This guide is written for beginners and assumes nothing is set up yet.

## What you will install
- Docker Desktop (to run the simulator in containers)
- VS Code (recommended editor)
- Git (to download the repo)

## 0) Install prerequisites (Windows)
### A) Install Docker Desktop
1) Download: https://www.docker.com/products/docker-desktop/
2) Install with defaults (Linux containers enabled).
3) Start Docker Desktop and wait until it says "Running".

### B) Install VS Code
1) Download: https://code.visualstudio.com/
2) Install with defaults.

### C) Install Git
1) Download: https://git-scm.com/downloads
2) Install with defaults.

## 1) Clone your repo
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

## 2) Verify required folders exist
Make sure this folder exists:
```
d:\RacerBot\f1tenth_gym_ros\f1tenth_gym\f1tenth_gym
```
If it does not exist, stop and ask for help before continuing.

## 3) Build the Docker image
```powershell
cd d:\RacerBot\f1tenth_gym_ros

docker compose build
```

This first build can take a long time.

## 4) Start the containers
```powershell
docker compose up -d
```

You should now have:
- f1tenth_gym_ros-sim-1
- f1tenth_gym_ros-novnc-1

## 5) Open the noVNC display
Open in your browser:
```
http://localhost:8080/vnc.html
```
Click **Connect**.

## 6) Launch the simulation
Run this from PowerShell:
```powershell
docker exec -d f1tenth_gym_ros-sim-1 bash -lc "source /sim_ws/.venv/bin/activate; source /opt/ros/humble/setup.bash; source /sim_ws/install/local_setup.bash; ros2 launch f1tenth_gym_ros gym_bridge_launch.py 2>&1 | tee /tmp/gym_bridge.log"
```

Optional: check logs
```powershell
docker exec f1tenth_gym_ros-sim-1 bash -lc "tail -n 50 /tmp/gym_bridge.log"
```

## 7) Connect Foxglove
1) Open https://app.foxglove.dev
2) Add a connection:
   - Connection type: Foxglove WebSocket
   - URL: ws://localhost:8765
3) Import the layout file:
   - d:\RacerBot\f1tenth_gym_ros\launch\gym_bridge_foxglove.json

## 8) Drive the car (WASD + arrows)
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

## Reset the car to origin
Use Foxglove:
1) In the 3D panel settings, set Fixed frame to `map`.
2) Use the **2D Pose Estimate** tool.
3) Click at the origin and drag to set heading.

## Troubleshooting
### Foxglove connection failed
- Make sure you selected **Foxglove WebSocket**, not ROS/rosbridge.
- Confirm port mapping exists:
  ```powershell
  docker port f1tenth_gym_ros-sim-1 8765
  ```
- Reconnect using ws://localhost:8765

### Build is slow
- First build is slow by design.
- Rebuilds are faster because BuildKit cache is enabled.

## Optional: Native Ubuntu (ROS 2 Humble)
If you want native Ubuntu instructions, see README.old.md.
