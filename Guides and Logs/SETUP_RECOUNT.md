# Setup Recount: F1TENTH Gym ROS2 Bridge

Date range: 2026-02-06 to 2026-02-07
Host OS: Windows
Runtime: Docker Desktop + noVNC
Branches: dev-humble for both repos

## Step-by-step recount (commands included)

1) Checked repository branches (both dev-humble):

```powershell
cd d:\RacerBot\f1tenth_gym_ros
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD

cd d:\RacerBot\f1tenth_gym_ros\f1tenth_gym\f1tenth_gym
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
```

1) First Docker build attempt (failed with nested gym path):

```powershell
cd d:\RacerBot\f1tenth_gym_ros
docker compose build
```

Error:

```text
ERROR: file:///sim_ws/src/f1tenth_gym_ros/f1tenth_gym does not appear to be a Python project
```

Fix applied in Dockerfile:

```text
pip install -e /sim_ws/src/f1tenth_gym_ros/f1tenth_gym/f1tenth_gym
```

1) Rebuild (failed with uv-build missing):

```powershell
docker compose build
```

Error:

```text
RuntimeError: uv-build was not properly installed
```

Fix applied in Dockerfile:

```text
pip3 install -U pip uv_build
pip3 install -e /sim_ws/src/f1tenth_gym_ros/f1tenth_gym/f1tenth_gym
```

1) Added build caching and smaller context:

- Dockerfile: BuildKit cache for apt and pip.
- Added .dockerignore.

1) Successful build:

```powershell
docker compose build
```

1) Start containers:

```powershell
docker compose up -d
```

1) Launch bridge (background + log capture):

```powershell
docker exec -d f1tenth_gym_ros-sim-1 bash -lc "source /sim_ws/.venv/bin/activate; source /opt/ros/humble/setup.bash; source /sim_ws/install/local_setup.bash; ros2 launch f1tenth_gym_ros gym_bridge_launch.py 2>&1 | tee /tmp/gym_bridge.log"
```

1) Foxglove connection error:

- Foxglove showed “Connection failed / Waiting for events…”.
- Bridge logs showed “Server listening on port 8765”.

Check logs:

```powershell
docker exec f1tenth_gym_ros-sim-1 bash -lc "tail -n 50 /tmp/gym_bridge.log"
```

1) Exposed WebSocket port in docker-compose.yml:

```yaml
ports:
  - "8765:8765"
```

1) Restart containers:

```powershell
docker compose up -d
```

1) Verify port mapping and connectivity:

```powershell
docker compose ps
docker port f1tenth_gym_ros-sim-1 8765
Test-NetConnection -ComputerName localhost -Port 8765
```

1) Connect Foxglove:

- Connection type: Foxglove WebSocket
- URL: ws://localhost:8765

1) Added WASD/arrow teleop node:

- Created f1tenth_gym_ros/wasd_teleop.py
- Added console script in setup.py:
  - wasd_teleop = f1tenth_gym_ros.wasd_teleop:main

1) Rebuild ROS workspace in container:

```powershell
docker exec f1tenth_gym_ros-sim-1 bash -lc "source /opt/ros/humble/setup.bash; cd /sim_ws; colcon build --symlink-install"
```

1) README updated with new beginner workflow and troubleshooting.

## Final working flow

- Build image: docker compose build
- Start containers: docker compose up -d
- Launch bridge: ros2 launch f1tenth_gym_ros gym_bridge_launch.py
- Connect Foxglove: ws://localhost:8765 (Foxglove WebSocket)
- Teleop: ros2 run f1tenth_gym_ros wasd_teleop

## Commands run (exact sequence)

```powershell
cd d:\RacerBot\f1tenth_gym_ros
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD

cd d:\RacerBot\f1tenth_gym_ros\f1tenth_gym\f1tenth_gym
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD

cd d:\RacerBot\f1tenth_gym_ros
docker compose build
docker compose build

docker compose up -d

docker exec -d f1tenth_gym_ros-sim-1 bash -lc "source /sim_ws/.venv/bin/activate; source /opt/ros/humble/setup.bash; source /sim_ws/install/local_setup.bash; ros2 launch f1tenth_gym_ros gym_bridge_launch.py 2>&1 | tee /tmp/gym_bridge.log"

docker exec f1tenth_gym_ros-sim-1 bash -lc "tail -n 50 /tmp/gym_bridge.log"

docker compose up -d

docker compose ps
docker port f1tenth_gym_ros-sim-1 8765
Test-NetConnection -ComputerName localhost -Port 8765

docker exec f1tenth_gym_ros-sim-1 bash -lc "source /opt/ros/humble/setup.bash; cd /sim_ws; colcon build --symlink-install"
```
