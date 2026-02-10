# Copilot Instructions

## Project context
- ROS 2 bridge for the F1TENTH Gym environment.
- Primary runtime is Docker Desktop + noVNC; ROS 2 Humble inside container.
- `f1tenth_gym` is a nested repo at `f1tenth_gym/f1tenth_gym`.
- Default branch for this fork is `main`; upstream references may mention `dev-humble`.
- Host OS is often Windows; mac builds may use amd64 emulation.

## How the system is expected to run
- Containers:
	- `sim` runs the ROS 2 stack and bridge.
	- `novnc` provides a browser-based GUI.
- Foxglove connects to `ws://localhost:8765` (WebSocket, not rosbridge).
- noVNC is at `http://localhost:8080/vnc.html`.
- Build and run is normally:
	- `docker compose build`
	- `docker compose up -d`
	- `docker exec -d f1tenth_gym_ros-sim-1 bash -lc "source /sim_ws/.venv/bin/activate; source /opt/ros/humble/setup.bash; source /sim_ws/install/local_setup.bash; ros2 launch f1tenth_gym_ros gym_bridge_launch.py 2>&1 | tee /tmp/gym_bridge.log"`

## Repo layout (key files)
- `Dockerfile`: builds the ROS 2 image, installs gym in venv and system Python, runs `colcon build`.
- `docker-compose.yml`: runs `sim` + `novnc`; should publish port `8765` and `8080`.
- `f1tenth_gym_ros/gym_bridge.py`: main ROS bridge node.
- `f1tenth_gym_ros/wasd_teleop.py`: custom WASD/arrow teleop node.
- `setup.py`: registers console scripts (`gym_bridge`, `wasd_teleop`).
- `config/sim.yaml`: simulation configuration (map, topics, teleop toggles).
- `launch/`: launch files and Foxglove layout JSON.
- Docs/logs:
	- `README.md` (primary user guide)
	- `README.old.md` (legacy instructions)
	- `SETUP_LOG.md`, `SETUP_GUIDE.md`, `SETUP_RECOUNT.md`, `CHANGES_SUMMARY.md`

## Build pitfalls and fixes
- Gym repo is nested; editable install path must be:
	- `/sim_ws/src/f1tenth_gym_ros/f1tenth_gym/f1tenth_gym`
- `ModuleNotFoundError: f1tenth_gym` can occur if system Python lacks the gym install.
- On mac arm64, PyQt6 builds from source and fails; the current workaround is forcing `linux/amd64` in compose.
- Foxglove connection issues are usually due to wrong connection type or missing port 8765 mapping.

## Development guidelines
- Prefer minimal, incremental changes; avoid unrelated refactors.
- Keep instructions copy/paste friendly for beginners.
- Default to Docker workflow unless explicitly asked for native Ubuntu.
- Use ASCII only in files unless existing content requires Unicode.
- Avoid removing `<details>` blocks in README; they are used to reduce visible length.
- When changing runtime behavior, update README and SETUP_LOG/SETUP_GUIDE.

## Docker and build
- `docker-compose.yml` forces `linux/amd64` for mac compatibility.
- `Dockerfile` installs dependencies and builds workspace with `colcon`.
- When adding Python code, ensure it is installed via `setup.py` console scripts if it needs to be runnable with `ros2 run`.
- After code changes, run `docker exec f1tenth_gym_ros-sim-1 bash -lc "source /opt/ros/humble/setup.bash; cd /sim_ws; colcon build --symlink-install"`.

## Files and conventions
- ROS bridge node: `f1tenth_gym_ros/gym_bridge.py`.
- Teleop node: `f1tenth_gym_ros/wasd_teleop.py`.
- Config: `config/sim.yaml`.
- Launch files: `launch/`.
- Documentation: `README.md` is user-facing and should stay beginner-friendly.

## Common pitfalls
- The nested gym repo path must be used for editable installs.
- Foxglove must connect via WebSocket to `ws://localhost:8765`.
- If a change requires rebuild, call out `colcon build --symlink-install` in the container.
- On mac, `docker compose build` will use amd64 emulation (slower, but avoids PyQt6 build errors).

## When editing docs
- Use Markdown links instead of bare URLs.
- Keep steps explicit and ordered.
- Use `<details>` blocks only where the README already uses them.
- Keep Windows paths as `d:\RacerBot\...` in PowerShell examples and Unix paths in bash examples.
