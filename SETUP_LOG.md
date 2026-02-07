# F1TENTH Gym ROS2 (Docker Desktop + noVNC) Setup Log

Date: 2026-02-06
Host OS: Windows
Repo: f1tenth/f1tenth_gym_ros (branch dev-humble)

## Goal
Set up and run the ROS2 gym bridge using Docker Desktop with noVNC.

## Actions performed
1. Verified repo layout and container configuration:
   - docker-compose.yml defines `sim` and `novnc` services.
   - Dockerfile builds ROS 2 Humble image and installs f1tenth_gym in a venv.
2. Chosen runtime path: Docker Desktop + noVNC (no GPU passthrough).
3. Built image, started containers, launched gym bridge.
4. Connected Foxglove using WebSocket.

## Commands executed
> Note: These commands are run from the repository root (this folder).

```powershell
# Build and start the containers
cd d:\RacerBot\f1tenth_gym_ros

docker compose up -d --build
```

Build completed successfully and the image `f1tenth_gym_ros:latest` was created.

```powershell
# Start services after successful build
docker compose up -d
```

Containers created:
- `f1tenth_gym_ros-sim-1`
- `f1tenth_gym_ros-novnc-1`

## Issues encountered and fixes
### Build error: editable install path not found
Error during Docker build:
```
ERROR: file:///sim_ws/src/f1tenth_gym_ros/f1tenth_gym does not appear to be a Python project
```
Root cause: the Python package lives at `f1tenth_gym/f1tenth_gym` (nested).

Fix applied:
- Updated the editable install path in Dockerfile.
  - From: `/sim_ws/src/f1tenth_gym_ros/f1tenth_gym`
  - To: `/sim_ws/src/f1tenth_gym_ros/f1tenth_gym/f1tenth_gym`

Re-run:
```powershell
docker compose up -d --build
```

### Build error: uv-build not properly installed
Error during Docker build:
```
RuntimeError: uv-build was not properly installed
```
Root cause: a second editable install using system Python attempted to run the PEP 517 backend without uv-build.

Fix applied:
- Installed `uv_build` in system Python and re-enabled the system install so ROS nodes can import `f1tenth_gym`.

Re-run:
```powershell
docker compose build
docker compose up -d
```

## Build speed improvements applied
- Enabled BuildKit cache for APT and pip in Dockerfile.
- Added `.dockerignore` to reduce build context size.

### Runtime error: `ModuleNotFoundError: No module named 'f1tenth_gym'`
The ROS2 node used system Python, which did not have `f1tenth_gym` installed.

Fix applied:
- Updated [Dockerfile](Dockerfile) to also install `f1tenth_gym` into system Python:
  - `pip3 install -e /sim_ws/src/f1tenth_gym_ros/f1tenth_gym/f1tenth_gym`

Rebuild and restart:
```powershell
docker compose down
docker compose up -d --build
```

## How to access the GUI
- Open http://localhost:8080/vnc.html
- Click **Connect** to access the virtual display.

## How to launch the simulation (inside the sim container)
```powershell
# Open a shell inside the sim container

docker exec -it f1tenth_gym_ros-sim-1 /bin/bash
```

Then inside the container:
```bash
source $HOME/sim_ws/.venv/bin/activate
source /opt/ros/humble/setup.bash
source /sim_ws/install/local_setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

Or launched in detached mode from the host:
```powershell
docker exec -d f1tenth_gym_ros-sim-1 bash -lc "source /sim_ws/.venv/bin/activate; source /opt/ros/humble/setup.bash; source /sim_ws/install/local_setup.bash; ros2 launch f1tenth_gym_ros gym_bridge_launch.py"
```

Final runtime command used (with log capture):
```powershell
docker exec -d f1tenth_gym_ros-sim-1 bash -lc "source /sim_ws/.venv/bin/activate; source /opt/ros/humble/setup.bash; source /sim_ws/install/local_setup.bash; ros2 launch f1tenth_gym_ros gym_bridge_launch.py 2>&1 | tee /tmp/gym_bridge.log"
```

Check logs for the Foxglove WebSocket URL:
```powershell
docker logs --tail 50 f1tenth_gym_ros-sim-1
```

## Visualization
- Foxglove is recommended. Use the WebSocket URL printed in the terminal.
- Connection type must be **Foxglove WebSocket** (not ROS/rosbridge).
- Port mapping added: `8765:8765` in docker-compose.yml.
- Import the layout file: `launch/gym_bridge_foxglove.json`.

## Notes / Known limitations
- GPU passthrough is not used in this setup.
- If you change configuration in `config/sim.yaml`, rebuild with:
  ```bash
  colcon build
  ```

## Runtime issue: Foxglove connection failed
Symptoms:
- Foxglove showed “Connection failed” and “Waiting for events…”.

Fixes applied:
- Exposed port 8765 from the sim container in docker-compose.yml.
- Connected using **Foxglove WebSocket** to ws://localhost:8765.

## Next steps
- Run your ROS2 nodes in another terminal or tmux session in the container.
