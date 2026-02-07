# F1TENTH Gym ROS2 (Docker Desktop + noVNC) Step-by-Step Guide

This guide sets up the F1TENTH Gym ROS2 bridge on Windows using Docker Desktop and noVNC.

## 1) Prerequisites
- Docker Desktop installed and running (Linux containers enabled).
- Git installed.
- A stable internet connection (first build is large).

## 2) Clone the repositories
Open PowerShell and run:

```powershell
# Create a workspace folder (optional)
mkdir d:\RacerBot
cd d:\RacerBot

# Clone the ROS2 bridge

git clone -b dev-humble https://github.com/f1tenth/f1tenth_gym_ros.git

# Clone the Gym repo inside the bridge repo
cd d:\RacerBot\f1tenth_gym_ros

git clone -b dev-humble https://github.com/f1tenth/f1tenth_gym.git
```

The Gym repo must end up at:
- d:\RacerBot\f1tenth_gym_ros\f1tenth_gym\f1tenth_gym

## 3) Build the Docker image
From the repo root:

```powershell
cd d:\RacerBot\f1tenth_gym_ros

docker compose build
```

This may take a long time on first build.

## 4) Start containers

```powershell
docker compose up -d
```

Expected containers:
- f1tenth_gym_ros-sim-1
- f1tenth_gym_ros-novnc-1

## 5) Open the noVNC display
- Open http://localhost:8080/vnc.html
- Click **Connect**

## 6) Launch the simulation
Run the bridge in the sim container:

```powershell
# Launch in background with log capture

docker exec -d f1tenth_gym_ros-sim-1 bash -lc "source /sim_ws/.venv/bin/activate; source /opt/ros/humble/setup.bash; source /sim_ws/install/local_setup.bash; ros2 launch f1tenth_gym_ros gym_bridge_launch.py 2>&1 | tee /tmp/gym_bridge.log"
```

Check logs if needed:

```powershell
docker exec f1tenth_gym_ros-sim-1 bash -lc "tail -n 50 /tmp/gym_bridge.log"
```

## 7) Connect Foxglove
1) Open https://app.foxglove.dev
2) Add a connection:
   - Connection type: **Foxglove WebSocket**
   - URL: `ws://localhost:8765`
3) Import the layout file:
   - d:\RacerBot\f1tenth_gym_ros\launch\gym_bridge_foxglove.json

## 8) Optional: Keyboard teleop
1) Edit [config/sim.yaml](config/sim.yaml): set `kb_teleop: true`.
2) Run teleop in another terminal:

```powershell
# Open a shell inside the sim container

docker exec -it f1tenth_gym_ros-sim-1 /bin/bash
```

Then:

```bash
source /opt/ros/humble/setup.bash
source /sim_ws/install/local_setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

## Issues and solutions
### Build error: editable install path not found
**Error:**
```
ERROR: file:///sim_ws/src/f1tenth_gym_ros/f1tenth_gym does not appear to be a Python project
```
**Fix:** Use the nested path:
- `/sim_ws/src/f1tenth_gym_ros/f1tenth_gym/f1tenth_gym`

### Build error: uv-build not properly installed
**Error:**
```
RuntimeError: uv-build was not properly installed
```
**Fix:** Install uv_build in system Python and reinstall f1tenth_gym in system Python.

### Runtime error: ModuleNotFoundError: No module named 'f1tenth_gym'
**Fix:** Install f1tenth_gym into system Python inside the image (already handled in the Dockerfile).

### Foxglove connection failed
**Symptoms:**
- “Connection failed”
- “Waiting for events…”

**Fixes:**
- Make sure port 8765 is published in docker-compose.yml.
- Connect using **Foxglove WebSocket** (not ROS/rosbridge).

## Build speed tips
- BuildKit caching is enabled for APT and pip.
- `.dockerignore` reduces build context size.
- Rebuild with `docker compose build` to reuse caches.

## Verify services
```powershell
# Containers

docker compose ps

# Port mapping

docker port f1tenth_gym_ros-sim-1 8765
```
